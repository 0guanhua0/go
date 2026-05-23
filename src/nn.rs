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
    pub fn new(device: Device, batch_size: usize, model_path: &str) -> Self {
        let (tx, rx) = mpsc::channel::<(Tensor, mpsc::Sender<(Tensor, Tensor)>)>();
        let path = std::path::Path::new(model_path);
        let model = CModule::load_on_device(path, device).unwrap();
        let id = path.file_stem().unwrap().to_string_lossy().to_string();

        let model_id = Arc::new(Mutex::new(id));

        let thread_handle = thread::spawn(move || {
            let mut queue = Vec::with_capacity(batch_size);

            loop {
                if let Ok(req) = rx.recv() {
                    queue.push(req);

                    while queue.len() < batch_size {
                        if let Ok(req) = rx.try_recv() {
                            queue.push(req);
                        } else {
                            break;
                        }
                    }
                } else {
                    break;
                }

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

                let policy = p.split(1, 0);
                let value = v.split(1, 0);
                for (i, (_, response_tx)) in queue.drain(..).enumerate() {
                    let p = policy[i].squeeze_dim(0);
                    let v = value[i].squeeze_dim(0);
                    let _ = response_tx.send((p, v));
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

    pub fn evaluate(&self, input: Tensor) -> (Tensor, Tensor) {
        let (tx, rx) = mpsc::channel();
        if let Some(sender) = &self.sender {
            sender.send((input, tx)).unwrap();
        }
        rx.recv().unwrap()
    }
}
