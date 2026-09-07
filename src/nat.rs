use std::{fmt, num::NonZeroU64};

use num_bigint::BigUint;

/// Arbitrary-precision natural number that is always positive.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PositiveNat(BigUint);

impl PositiveNat {
    pub fn one() -> Self {
        Self(BigUint::from(1_u8))
    }

    pub fn successor(&self) -> Self {
        Self(&self.0 + 1_u8)
    }

    pub fn is_one(&self) -> bool {
        self.0.bits() == 1
    }

    pub fn predecessor(&self) -> Option<Self> {
        (!self.is_one()).then(|| Self(&self.0 - 1_u8))
    }

    pub fn shifted(&self, bits: usize) -> Self {
        Self(&self.0 << bits)
    }

    pub fn ln(&self) -> f64 {
        let bits = self.0.bits();
        if bits <= 53 {
            return (self.0.to_u64_digits()[0] as f64).ln();
        }

        let shift = (bits - 53) as usize;
        let top = (&self.0 >> shift).to_u64_digits()[0];
        (top as f64).ln() + shift as f64 * std::f64::consts::LN_2
    }
}

impl From<NonZeroU64> for PositiveNat {
    fn from(value: NonZeroU64) -> Self {
        Self(BigUint::from(value.get()))
    }
}

impl fmt::Display for PositiveNat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successor_crosses_machine_integer_width() {
        let max = PositiveNat::from(NonZeroU64::new(u64::MAX).unwrap());
        assert_eq!(max.successor().to_string(), "18446744073709551616");
    }
}
