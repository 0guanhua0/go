use std::sync::mpsc;
use std::thread;
use tch::{Device, Kind, Tensor, nn};

struct AlphaGoZero {
    conv_block: nn::SequentialT,
    res_tower: nn::SequentialT,
    policy_conv: nn::Conv2D,
    policy_bn: nn::BatchNorm,
    policy_l: nn::Linear,
    value_conv: nn::Conv2D,
    value_bn: nn::BatchNorm,
    value_l1: nn::Linear,
    value_l2: nn::Linear,
}

impl AlphaGoZero {
    fn new(
        p: &nn::Path,
        board: i64,
        input_planes: i64,
        conv_filter: i64,
        res_block_cnt: i64,
    ) -> Self {
        let mut conv_block = nn::seq_t();
        conv_block = conv_block.add(nn::conv2d(
            p / "conv_block" / "0",
            input_planes,
            conv_filter,
            3,
            nn::ConvConfig {
                stride: 1,
                padding: 1,
                bias: false,
                ..Default::default()
            },
        ));
        conv_block = conv_block.add(nn::batch_norm2d(
            p / "conv_block" / "1",
            conv_filter,
            Default::default(),
        ));
        conv_block = conv_block.add_fn(|x| x.relu());

        let mut res_tower = nn::seq_t();
        let res_tower_path = p / "res_tower";
        for i in 0..res_block_cnt {
            let p_res = &res_tower_path / i;
            let conv1 = nn::conv2d(
                &p_res / "conv1",
                conv_filter,
                conv_filter,
                3,
                nn::ConvConfig {
                    stride: 1,
                    padding: 1,
                    bias: false,
                    ..Default::default()
                },
            );
            let bn1 = nn::batch_norm2d(&p_res / "bn1", conv_filter, Default::default());
            let conv2 = nn::conv2d(
                &p_res / "conv2",
                conv_filter,
                conv_filter,
                3,
                nn::ConvConfig {
                    stride: 1,
                    padding: 1,
                    bias: false,
                    ..Default::default()
                },
            );
            let bn2 = nn::batch_norm2d(&p_res / "bn2", conv_filter, Default::default());
            res_tower = res_tower.add_fn_t(move |x, train| {
                let identity = x;
                let x = x.apply_t(&conv1, train).apply_t(&bn1, train).relu();
                let x = x.apply_t(&conv2, train).apply_t(&bn2, train);
                (x + identity).relu()
            });
        }

        let policy_conv = nn::conv2d(
            p / "policy_conv",
            conv_filter,
            2,
            1,
            nn::ConvConfig {
                stride: 1,
                bias: false,
                ..Default::default()
            },
        );
        let policy_bn = nn::batch_norm2d(p / "policy_bn", 2, Default::default());
        let policy_l = nn::linear(
            p / "policy_l",
            2 * board * board,
            board * board + 1,
            Default::default(),
        );

        let value_conv = nn::conv2d(
            p / "value_conv",
            conv_filter,
            1,
            1,
            nn::ConvConfig {
                stride: 1,
                bias: false,
                ..Default::default()
            },
        );
        let value_bn = nn::batch_norm2d(p / "value_bn", 1, Default::default());
        let value_l1 = nn::linear(
            p / "value_l1",
            board * board,
            conv_filter,
            Default::default(),
        );
        let value_l2 = nn::linear(p / "value_l2", conv_filter, 1, Default::default());

        Self {
            conv_block,
            res_tower,
            policy_conv,
            policy_bn,
            policy_l,
            value_conv,
            value_bn,
            value_l1,
            value_l2,
        }
    }

    fn forward_t(&self, x: &Tensor, train: bool) -> (Tensor, Tensor) {
        let x = x.apply_t(&self.conv_block, train);
        let x = x.apply_t(&self.res_tower, train);

        let policy = x
            .apply_t(&self.policy_conv, train)
            .apply_t(&self.policy_bn, train)
            .relu();
        let policy = policy.view([x.size()[0], -1]);
        let policy = policy.apply(&self.policy_l);

        let value = x
            .apply_t(&self.value_conv, train)
            .apply_t(&self.value_bn, train)
            .relu();
        let value = value.view([x.size()[0], -1]);
        let value = value.apply(&self.value_l1).relu();
        let value = value.apply(&self.value_l2).tanh();

        (policy, value)
    }
}

pub struct Batcher {
    sender: mpsc::Sender<(Tensor, mpsc::Sender<(Tensor, Tensor)>)>,
}

impl Batcher {
    pub fn new(
        device: Device,
        batch_size: usize,
        board: i64,
        input_planes: i64,
        conv_filter: i64,
        res_block: i64,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<(Tensor, mpsc::Sender<(Tensor, Tensor)>)>();

        thread::spawn(move || {
            let vs = nn::VarStore::new(device);
            let model = AlphaGoZero::new(&vs.root(), board, input_planes, conv_filter, res_block);

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

                let inputs: Vec<Tensor> = queue.iter().map(|(t, _)| t.shallow_clone()).collect();
                let batch_input = Tensor::cat(&inputs, 0).to(device);

                let (policy, value) = model.forward_t(&batch_input, false);

                let policies = policy.to_kind(Kind::Float).split(1, 0);
                let values = value.to_kind(Kind::Float).split(1, 0);

                for (i, (_, response_tx)) in queue.drain(..).enumerate() {
                    let p = policies[i].squeeze_dim(0);
                    let v = values[i].squeeze_dim(0);
                    let _ = response_tx.send((p, v));
                }
            }
        });

        Self { sender: tx }
    }

    pub fn evaluate(&self, input: Tensor) -> (Tensor, Tensor) {
        let (tx, rx) = mpsc::channel();
        self.sender.send((input, tx)).unwrap();
        rx.recv().unwrap()
    }
}
