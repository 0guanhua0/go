use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use tch::{CModule, Device, IValue, Tensor};

pub struct Batcher {
    sender: Option<mpsc::Sender<(Tensor, mpsc::Sender<(Tensor, Tensor)>)>>,
    model_id: Arc<Mutex<String>>,
    thread_handle: Option<thread::JoinHandle<()>>,
}

impl Drop for Batcher {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Batcher {
    pub fn new(device: Device, model_path: &str) -> Self {
        let (tx, rx) = mpsc::channel::<(Tensor, mpsc::Sender<(Tensor, Tensor)>)>();
        let path = std::path::Path::new(model_path);
        let model = CModule::load_on_device(path, device).unwrap();
        let id = path.file_stem().unwrap().to_string_lossy().to_string();
        let model_id = Arc::new(Mutex::new(id));
        let thread_handle = thread::spawn(move || {
            let mut queue = Vec::new();
            loop {
                if let Ok(req) = rx.recv() {
                    queue.push(req);
                    while let Ok(req) = rx.try_recv() {
                        queue.push(req);
                    }
                } else {
                    break;
                }
                let batch_sizes: Vec<i64> = queue.iter().map(|(t, _)| t.size()[0]).collect();
                let input: Vec<Tensor> = queue.iter().map(|(t, _)| t.shallow_clone()).collect();
                let input = Tensor::cat(&input, 0).to(device);
                let output = model.forward_is(&[IValue::from(input)]).unwrap();
                let mut e = match output {
                    IValue::Tuple(e) => e,
                    _ => panic!(),
                };
                let (p, v) = match (e.remove(0), e.remove(0)) {
                    (IValue::Tensor(p), IValue::Tensor(v)) => (p, v),
                    _ => panic!(),
                };
                let mut offset: i64 = 0;
                for ((_, response_tx), &bs) in queue.drain(..).zip(batch_sizes.iter()) {
                    let pi = p.narrow(0, offset, bs);
                    let vi = v.narrow(0, offset, bs);
                    offset += bs;
                    let _ = response_tx.send((pi, vi));
                }
            }
        });

        Self {
            sender: Some(tx),
            model_id,
            thread_handle: Some(thread_handle),
        }
    }

    pub fn model_id(&self) -> String {
        self.model_id.lock().unwrap().clone()
    }

    pub fn eval(&self, input: Tensor) -> (Tensor, Tensor) {
        let (tx, rx) = mpsc::channel();
        if let Some(sender) = &self.sender {
            sender.send((input, tx)).unwrap();
        }
        rx.recv().unwrap()
    }
}
