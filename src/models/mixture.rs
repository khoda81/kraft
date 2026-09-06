use crate::{Distribution, Model};

pub struct Mixture<A, B> {
    pub a: A,
    pub b: B,
    pub logit: f64,
}

impl<A, B> Mixture<A, B> {
    fn new(a: A, b: B) -> Self {
        Self { a, b, logit: 0.0 }
    }
}

pub struct MixturePrediction<'a, A, B> {
    pub a: &'a A,
    pub b: &'a B,
    pub logit: f64,
}

impl<A, B, T> Distribution<T> for MixturePrediction<'_, A, B>
where
    A: Distribution<T>,
    B: Distribution<T>,
{
    fn ln_prob(&self, observation: &T) -> f64 {
        let a_ln_prob = self.a.ln_prob(observation);
        let b_ln_prob = self.b.ln_prob(observation);

        // p(x) = p(a) * sig(logit) + p(b) * (1 - sig(logit))
        (a_ln_prob.exp() + (b_ln_prob + self.logit).exp()) / (self.logit.exp() + 1.0)
    }
}

impl<A, B, T> Model<T> for Mixture<A, B>
where
    A: Distribution<T>,
    B: Distribution<T>,
{
    fn predict(&self) -> MixturePrediction<'_, A, B> {
        todo!()
    }

    fn observe(&mut self, observation: T) {
        todo!()
    }
}
