//! NeuroFlow is neural networks (and deep learning of course) Rust crate.
//! It relies on three pillars: speed, reliability, and speed again.
//!
//! Let's better check some examples.
//!
//! # Examples
//!
//! Here we are going to approximate very simple function `0.5*sin(e^x) - cos(e^(-x))`.
//!
//! ```rust
//!
//! use neuroflow::FeedForward;
//! use neuroflow::data::DataSet;
//! use neuroflow::activators::Type::Tanh;
//!
//!
//!  /*
//!      Define neural network with 1 neuron in input layers. Network contains 4 hidden layers.
//!      And, such as our function returns single value, it is reasonable to have 1 neuron in
//!      the output layer.
//!  */
//!  let mut nn = FeedForward::new(&[1, 7, 8, 8, 7, 1]);
//!
//!  /*
//!      Define DataSet.
//!
//!      DataSet is the Type that significantly simplifies work with neural network.
//!      Majority of its functionality is still under development :(
//!  */
//!  let mut data: DataSet = DataSet::new();
//!  let mut i = -3.0;
//!
//!  // Push the data to DataSet (method push accepts two slices: input data and expected output)
//!  while i <= 2.5 {
//!      data.push(&[i], &[0.5*(i.exp().sin()) - (-i.exp()).cos()]);
//!      i += 0.05;
//!  }
//!
//!  // Here, we set necessary parameters and train neural network
//!  // by our DataSet with 50 000 iterations
//!  nn.activation(Tanh)
//!      .learning_rate(0.01)
//!      .train(&data, 50_000);
//!
//!  let mut res;
//!
//!  // Let's check the result
//!  i = 0.0;
//!  while i <= 0.3{
//!      res = nn.calc(&[i])[0];
//!      println!("for [{:.3}], [{:.3}] -> [{:.3}]", i, 0.5*(i.exp().sin()) - (-i.exp()).cos(), res);
//!      i += 0.07;
//!  }
//! ```
//!
//! You don't need to lose your so hardly trained network, my friend! For those there are
//! functions for saving and loading of neural networks to and from file. They are
//! located in the `neuroflow::io` module.
//!
//! ```rust
//! # use neuroflow::FeedForward;
//! use neuroflow::io;
//! # let mut nn = FeedForward::new(&[1, 7, 8, 8, 7, 1]);
//!  /*
//!     In order to save neural network into file call function save from neuroflow::io module.
//!
//!     First argument is link on the saving neural network;
//!     Second argument is path to the file.
//! */
//! io::save(&mut nn, "test.flow").unwrap();
//!
//! /*
//!     After we have saved the neural network to the file we can restore it by calling
//!     of load function from neuroflow::io module.
//!
//!     We must specify the type of new_nn variable.
//!     The only argument of load function is the path to file containing
//!     the neural network
//! */
//! let mut new_nn: FeedForward = io::load("test.flow").unwrap();
//! ```
//!
//! We did say a little words about `DataSet` structure. It deserves to be considered
//! more precisely.
//!
//! Simply saying `DataSet` is just container for your input vectors and desired output to them,
//! but with additional functionality.
//!
//! ```rust
//! use std::path::Path;
//! use neuroflow::data::DataSet;
//!
//! // You can create empty DataSet calling its constructor new
//! let mut d1 = DataSet::new();
//!
//! // To push new data to DataSet instance call push method
//! d1.push(&[0.1, 0.2], &[1.0, 2.3]);
//! d1.push(&[0.05, 0.01], &[0.5, 1.1]);
//!
//! // You can load data from csv file
//! let p = "file.csv";
//! if Path::new(p).exists(){
//!     let mut d2 = DataSet::from_csv(p); // Easy, eah?
//! }
//!
//! // You can round all DataSet elements with precision
//! d1.round(2); // 2 is the amount of digits after point
//!
//! // Also, it is possible to get some statistical information.
//! // For current version it is possible to get only mean values (by each dimension or by
//! // other words each column in vector) of input vector and desired output vector
//! let (x, y) = d1.mean();
//!
//! ```
//!

#![no_std]

extern crate heapless;
extern crate serde;

use heapless::Vec;
use serde::{Deserialize, Serialize};

/// Struct `Layer` represents single layer of network.
/// It is private and should not be used directly.
#[derive(Serialize, Deserialize, Debug)]
struct Layer {
    v: Vec<f32, 8>,
    y: Vec<f32, 8>,
    delta: Vec<f32, 8>,
    prev_delta: Vec<f32, 8>,
    w: Vec<Vec<f32, 8>, 8>,
}

/// Feed Forward (multilayer perceptron) neural network that is trained
/// by back propagation algorithm.
/// You can use it for approximation and classification tasks as well.
///
/// # Examples
///
/// In order to create `FeedForward` instance call its constructor `new`.
///
/// The constructor accepts slice as an argument. This slice determines
/// the architecture of network.
/// First element in slice is amount of neurons in input layer
/// and the last one is amount of neurons in output layer.
/// Denote, that vector of input data must have the equal length as input
/// layer of FeedForward neural network (the same is for expected output vector).
///
/// ```rust
/// use neuroflow::FeedForward;
///
/// let mut nn = FeedForward::new(&[1, 3, 2]);
/// ```
///
/// Then you can train your network simultaneously via `fit` method:
///
/// ```rust
/// # use neuroflow::FeedForward;
/// # let mut nn = FeedForward::new(&[1, 3, 2]);
/// nn.fit(&[1.2], &[0.2, 0.8]);
/// ```
///
/// Or to use `train` method with `neuroflow::data::DataSet` struct:
///
/// ```rust
/// # use neuroflow::FeedForward;
/// # let mut nn = FeedForward::new(&[1, 3, 2]);
/// use neuroflow::data::DataSet;
///
/// let mut data = DataSet::new();
/// data.push(&[1.2], &[1.3, -0.2]);
/// nn.train(&data, 30_000); // 30_000 is iterations count
/// ```
///
/// It is possible to set parameters of network:
/// ```rust
/// # use neuroflow::FeedForward;
/// # let mut nn = FeedForward::new(&[1, 3, 2]);
/// nn.learning_rate(0.1)
///   .momentum(0.05)
///   .activation(neuroflow::activators::Type::Tanh);
/// ```
///
/// Call method `calc` in order to calculate value by your(already trained) network:
///
/// ```rust
/// # use neuroflow::FeedForward;
/// # let mut nn = FeedForward::new(&[1, 3, 2]);
/// let d: Vec<f32> = nn.calc(&[1.02]).to_vec();
/// ```
///
#[derive(Serialize, Deserialize)]
pub struct FeedForward {
    layers: Vec<Layer, 8>,
    pub learn_rate: f32,
    pub momentum: f32,
    pub error: f32,
    pub act_type: ActivatorType,
}

impl Layer {
    fn new<F>(amount: i32, input: i32, rand: F) -> Layer
    where
        F: Fn() -> f32,
    {
        let mut nl = Layer {
            v: Vec::new(),
            y: Vec::new(),
            delta: Vec::new(),
            prev_delta: Vec::new(),
            w: Vec::new(),
        };
        let mut v;
        for _ in 0..amount {
            nl.y.push(0.0).unwrap();
            nl.delta.push(0.0).unwrap();
            nl.v.push(0.0).unwrap();

            v = Vec::new();
            for _ in 0..input + 1 {
                v.push(2f32 * rand() - 1f32).unwrap();
            }

            nl.w.push(v).unwrap();
        }
        return nl;
    }
}

impl FeedForward {
    /// The constructor of `FeedForward` struct
    ///
    /// * `architecture: &[i32]` - the architecture of network where each
    /// element in slice represents amount of neurons in this layer.
    /// First element in slice is amount of neurons in input layer
    /// and the last one is amount of neurons in output layer.
    /// Denote, that vector of input data must have the equal length as input
    /// layer of FeedForward neural network (the same is for expected output vector).
    ///
    /// * `return` - `FeedForward` struct
    /// # Example
    ///
    /// ```rust
    /// use neuroflow::FeedForward;
    /// let mut nn = FeedForward::new(&[1, 3, 2]);
    /// ```
    ///
    pub fn new<F>(architecture: &[i32], rand: F) -> FeedForward
    where
        F: Fn() -> f32,
    {
        let mut nn = FeedForward {
            learn_rate: 0.1,
            momentum: 0.1,
            error: 0.0,
            layers: Vec::new(),
            act_type: ActivatorType::Tanh,
        };

        for i in 1..architecture.len() {
            nn.layers
                .push(Layer::new(architecture[i], architecture[i - 1], &rand))
                .unwrap();
        }

        return nn;
    }

    fn forward(&mut self, x: &[f32]) {
        let mut sum: f32;

        for j in 0..self.layers.len() {
            if j == 0 {
                for i in 0..self.layers[j].v.len() {
                    sum = self.layers[j].w[i][0];
                    for k in 0..x.len() {
                        sum += self.layers[j].w[i][k + 1] * x[k];
                    }
                    self.layers[j].v[i] = sum;
                    self.layers[j].y[i] = self.act_type.func(sum);
                }
            } else if j == self.layers.len() - 1 {
                for i in 0..self.layers[j].v.len() {
                    sum = self.layers[j].w[i][0];
                    for k in 0..self.layers[j - 1].y.len() {
                        sum += self.layers[j].w[i][k + 1] * self.layers[j - 1].y[k];
                    }
                    self.layers[j].v[i] = sum;
                    self.layers[j].y[i] = sum;
                }
            } else {
                for i in 0..self.layers[j].v.len() {
                    sum = self.layers[j].w[i][0];
                    for k in 0..self.layers[j - 1].y.len() {
                        sum += self.layers[j].w[i][k + 1] * self.layers[j - 1].y[k];
                    }
                    self.layers[j].v[i] = sum;
                    self.layers[j].y[i] = self.act_type.func(sum);
                }
            }
        }
    }

    fn backward(&mut self, d: &[f32]) {
        let mut sum: f32;

        for j in (0..self.layers.len()).rev() {
            self.layers[j].prev_delta = self.layers[j].delta.clone();
            if j == self.layers.len() - 1 {
                self.error = 0.0;
                for i in 0..self.layers[j].y.len() {
                    self.layers[j].delta[i] =
                        (d[i] - self.layers[j].y[i]) * self.act_type.der(self.layers[j].v[i]);
                    let temp = d[i] - self.layers[j].y[i];
                    self.error += 0.5 * temp * temp;
                }
            } else {
                for i in 0..self.layers[j].delta.len() {
                    sum = 0.0;
                    for k in 0..self.layers[j + 1].delta.len() {
                        sum += self.layers[j + 1].delta[k] * self.layers[j + 1].w[k][i + 1];
                    }
                    self.layers[j].delta[i] = self.act_type.der(self.layers[j].v[i]) * sum;
                }
            }
        }
    }

    fn update(&mut self, x: &[f32]) {
        for j in 0..self.layers.len() {
            for i in 0..self.layers[j].w.len() {
                for k in 0..self.layers[j].w[i].len() {
                    if j == 0 {
                        self.layers[j].w[i][k] += self.learn_rate
                            * self.layers[j].delta[i]
                            * if k == 0 { 1.0 } else { x[k - 1] };
                    } else {
                        if k == 0 {
                            self.layers[j].w[i][k] += self.learn_rate * self.layers[j].delta[i];
                        } else {
                            self.layers[j].w[i][k] += self.learn_rate
                                * self.layers[j].delta[i]
                                * self.layers[j - 1].y[k - 1];
                        }
                    }
                    self.layers[j].w[i][k] += self.momentum * self.layers[j].prev_delta[i];
                }
            }
        }
    }

    /// Train neural network by bulked data.
    ///
    /// * `data: &T` - the link on data that implements `neuroflow::data::Extractable` trait;
    /// * `iterations: i64` - iterations count.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use neuroflow::FeedForward;
    /// # let mut nn = FeedForward::new(&[1, 3, 2]);
    /// let mut d = neuroflow::data::DataSet::new();
    /// d.push(&[1.2], &[1.3, -0.2]);
    /// nn.train(&d, 30_000);
    /// ```
    pub fn train<'a, F>(&mut self, rand: F, iterations: i64)
    where
        F: Fn() -> (&'a [f32], &'a [f32]),
    {
        for _ in 0..iterations {
            let (x, y) = rand();
            self.forward(&x);
            self.backward(&y);
            self.update(&x);
        }
    }

    /// Calculate the response by trained neural network.
    ///
    /// * `X: &[f32]` - slice of input data;
    /// * `return -> &[f32]` - slice of calculated data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use neuroflow::FeedForward;
    /// # let mut nn = FeedForward::new(&[1, 3, 2]);
    /// let v: &[f32] = nn.calc(&[1.02]).to_vec();
    /// ```
    #[allow(non_snake_case)]
    pub fn calc(&mut self, x: &[f32]) -> &[f32] {
        self.forward(x);
        &self.layers[self.layers.len() - 1].y
    }
}

/// Determine types of activation functions contained in this module.
#[allow(dead_code)]
#[derive(Serialize, Deserialize)]
pub enum ActivatorType {
    Sigmoid,
    Tanh,
    Relu,
}

impl ActivatorType {
    pub fn func(&self, x: f32) -> f32 {
        match self {
            Self::Sigmoid => 1.0 / (1.0 + libm::expf(x)),
            Self::Tanh => libm::tanhf(x),
            Self::Relu => f32::max(0.0, x),
        }
    }

    pub fn der(&self, x: f32) -> f32 {
        match self {
            Self::Sigmoid => self.func(x) * (1.0 - self.func(x)),
            Self::Tanh => {
                let temp = self.func(x);
                1.0 - temp * temp
            }
            Self::Relu => {
                if x <= 0.0 {
                    0.0
                } else {
                    1.0
                }
            }
        }
    }
}

pub mod estimators {
    /// # Widrow's rule of thumb
    /// This is an empirical rule that shows the size of training sample
    /// in order to get good generalization.
    /// ## Example
    /// For network architecture [2, 1] and allowed error 0.1 (10%)
    /// the size of training sample must exceed the amount of free
    /// network parameters in 10 times
    pub fn widrows(architecture: &[i32], allowed_error: f32) -> f32 {
        let mut s = architecture[0] * (architecture[0] + 1);

        for i in 1..architecture.len() {
            s += architecture[i] * architecture[i - 1] + architecture[i];
        }

        (s as f32) / allowed_error
    }
}
