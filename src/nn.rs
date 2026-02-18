use std::fs;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::Duration;
use tch::{CModule, Device, IValue, Tensor};

pub struct Batcher {
    sender: mpsc::Sender<(Tensor, mpsc::Sender<(Tensor, Tensor)>)>,
    model_id: Arc<Mutex<Option<String>>>,
}

impl Batcher {
    pub fn new(device: Device, batch_size: usize) -> Self {
        let (tx, rx) = mpsc::channel::<(Tensor, mpsc::Sender<(Tensor, Tensor)>)>();
        let model_id = Arc::new(Mutex::new(None));
        let model_id_clone = model_id.clone();

        thread::spawn(move || {
            let mut last_loaded_path = None;
            let mut model: Option<CModule> = None;

            let entries = fs::read_dir("models").unwrap();
            let newest = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_file())
                .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok());

            if let Some(entry) = newest {
                let path = entry.path();
                if let Ok(m) = CModule::load_on_device(&path, device) {
                    model = Some(m);
                    let id = path.file_stem().unwrap().to_str().unwrap().to_string();
                    *model_id_clone.lock().unwrap() = Some(id);
                    last_loaded_path = Some(path);
                }
            }

            let (reload_tx, reload_rx) = mpsc::channel();
            thread::spawn(move || {
                loop {
                    thread::sleep(Duration::from_secs(60));
                    let _ = reload_tx.send(());
                }
            });

            let mut queue = Vec::with_capacity(batch_size);

            loop {
                if reload_rx.try_recv().is_ok() {
                    if let Ok(entries) = fs::read_dir("models") {
                        let newest = entries
                            .filter_map(|e| e.ok())
                            .filter(|e| e.path().is_file())
                            .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok());

                        if let Some(entry) = newest {
                            let path = entry.path();
                            if Some(&path) != last_loaded_path.as_ref() {
                                if let Ok(m) = CModule::load_on_device(&path, device) {
                                    model = Some(m);
                                    let id =
                                        path.file_stem().unwrap().to_str().unwrap().to_string();
                                    *model_id_clone.lock().unwrap() = Some(id);
                                    last_loaded_path = Some(path);
                                }
                            }
                        }
                    }
                }

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

                if let Some(m) = &model {
                    let inputs: Vec<Tensor> =
                        queue.iter().map(|(t, _)| t.shallow_clone()).collect();
                    let batch_input = Tensor::cat(&inputs, 0).to(device);

                    let output = m.forward_is(&[IValue::from(batch_input)]).unwrap();
                    let IValue::Tuple(mut elements) = output else {
                        unreachable!()
                    };
                    let (IValue::Tensor(policy), IValue::Tensor(value)) =
                        (elements.remove(0), elements.remove(0))
                    else {
                        unreachable!()
                    };

                    let policies = policy.split(1, 0);
                    let values = value.split(1, 0);

                    for (i, (_, response_tx)) in queue.drain(..).enumerate() {
                        let p = policies[i].squeeze_dim(0);
                        let v = values[i].squeeze_dim(0);
                        let _ = response_tx.send((p, v));
                    }
                } else {
                    queue.clear();
                }
            }
        });

        Self {
            sender: tx,
            model_id,
        }
    }

    pub fn model_id(&self) -> String {
        self.model_id.lock().unwrap().clone().unwrap()
    }

    pub fn evaluate(&self, input: Tensor) -> (Tensor, Tensor) {
        let (tx, rx) = mpsc::channel();
        self.sender.send((input, tx)).unwrap();
        rx.recv().unwrap()
    }
}
