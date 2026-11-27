//! Fixed-cost exact averaging for the bounded kNN target roster.
use super::KnnError;
use std::cmp::Ordering;

// Every finite binary64 number is an integer multiple of 2^-1074.
// MAX uses 2098 magnitude bits in these units. Each of the separate
// positive and negative sums of at most 20,000 terms fits in 2113 bits.
const LIMBS: usize = 34;
const FRACTION_BITS: usize = 52;
const FRACTION_MASK: u64 = (1 << FRACTION_BITS) - 1;

struct Magnitude([u64; LIMBS]);

impl Default for Magnitude {
    fn default() -> Self {
        Self([0; LIMBS])
    }
}

impl Magnitude {
    fn add_word(&mut self, index: usize, value: u64) {
        if value == 0 {
            return;
        }
        let (sum, mut carry) = self.0[index].overflowing_add(value);
        self.0[index] = sum;
        for word in &mut self.0[index + 1..] {
            if !carry {
                break;
            }
            (*word, carry) = word.overflowing_add(1);
        }
        debug_assert!(!carry, "fixed accumulator must fit");
    }

    fn add_float_bits(&mut self, bits: u64) {
        let exponent = ((bits >> FRACTION_BITS) & 0x7ff) as usize;
        let hidden = u64::from(exponent != 0) << FRACTION_BITS;
        let significand = (bits & FRACTION_MASK) | hidden;
        let shift = exponent.saturating_sub(1);
        let index = shift / 64;
        let offset = shift % 64;
        self.add_word(index, significand << offset);
        if offset != 0 {
            self.add_word(index + 1, significand >> (64 - offset));
        }
    }

    fn subtract(&mut self, smaller: &Self) {
        let mut borrow = false;
        for (word, &other) in self.0.iter_mut().zip(&smaller.0) {
            let (difference, p) = word.overflowing_sub(other);
            let (difference, q) = difference.overflowing_sub(u64::from(borrow));
            *word = difference;
            borrow = p || q;
        }
        debug_assert!(!borrow);
    }

    // Divide the exact integer by the small neighbor count, retaining the
    // remainder so the final f64 conversion cannot double-round.
    fn divide(&mut self, count: u64) -> u64 {
        if count.is_power_of_two() {
            let shift = count.trailing_zeros();
            if shift == 0 {
                return 0;
            }
            let remainder = self.0[0] & (count - 1);
            let mut high = 0;
            for word in self.0.iter_mut().rev() {
                let original = *word;
                *word = (original >> shift) | high;
                high = original << (64 - shift);
            }
            return remainder;
        }

        let mut remainder = 0;
        // Limbs above the numerator's first nonzero word contribute
        // neither to the quotient nor to the remainder.
        for word in self.0.iter_mut().rev().skip_while(|word| **word == 0) {
            let dividend = (u128::from(remainder) << 64) | u128::from(*word);
            let divisor = u128::from(count);
            *word = (dividend / divisor) as u64;
            remainder = (dividend % divisor) as u64;
        }
        remainder
    }

    fn bit_length(&self) -> usize {
        self.0
            .iter()
            .rposition(|&word| word != 0)
            .map_or(0, |i| 64 * (i + 1) - self.0[i].leading_zeros() as usize)
    }

    fn bit(&self, index: usize) -> bool {
        (self.0[index / 64] >> (index % 64)) & 1 != 0
    }

    // Are any bits below the given (exclusive) boundary nonzero?
    fn any_below(&self, boundary: usize) -> bool {
        let index = boundary / 64;
        let offset = boundary % 64;
        self.0[..index].iter().any(|&word| word != 0)
            || (self.0[index] & ((1_u64 << offset) - 1)) != 0
    }

    fn shifted_word(&self, shift: usize) -> u64 {
        let index = shift / 64;
        let offset = shift % 64;
        let hi = if offset == 0 {
            0
        } else {
            self.0[index + 1] << (64 - offset)
        };
        (self.0[index] >> offset) | hi
    }

    fn rounded_mean_bits(mut self, count: u64) -> u64 {
        let remainder = self.divide(count);
        let shift = self.bit_length().saturating_sub(FRACTION_BITS + 1);
        let mut retained = self.shifted_word(shift);
        let odd = retained & 1 != 0;
        let round_up = if shift == 0 {
            // The output lattice is integer units of 2^-1074, including
            // the normal/subnormal boundary. Compare the remainder exactly.
            2 * remainder > count || (2 * remainder == count && odd)
        } else {
            // The remainder from division contributes to sticky bits
            // even if all the integer quotient's discarded bits are zero.
            self.bit(shift - 1) && (self.any_below(shift - 1) || remainder != 0 || odd)
        };
        retained += u64::from(round_up);
        // Adding the retained significand incorporates its hidden bit into
        // the encoded exponent. A rounding carry promotes the binade.
        ((shift as u64) << FRACTION_BITS) + retained
    }
}

/// The caller supplies exactly `count` selected, finite neighbor targets.
/// Keep the existing max-target normalization acceptance boundary even
/// though the exact arithmetic no longer needs normalization.
pub(crate) fn checked_mean(
    values: impl Iterator<Item = f64> + Clone,
    count: usize,
) -> Result<f64, KnnError> {
    debug_assert!((1..=20_000).contains(&count));
    let scale = values.clone().map(f64::abs).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Ok(0.0);
    }
    if count == 1 {
        return Ok(values.clone().next().unwrap());
    }

    let mut positive = Magnitude::default();
    let mut negative = Magnitude::default();
    for value in values {
        if value != 0.0 && value / scale == 0.0 {
            return Err(KnnError::NumericOverflow);
        }
        if value.is_sign_negative() {
            negative.add_float_bits(value.to_bits());
        } else {
            positive.add_float_bits(value.to_bits());
        }
    }

    let (mut larger, smaller, sign) = match positive.0.iter().rev().cmp(negative.0.iter().rev()) {
        Ordering::Equal => return Ok(0.0),
        Ordering::Greater => (positive, negative, 0),
        Ordering::Less => (negative, positive, 1_u64 << 63),
    };
    larger.subtract(&smaller);
    let magnitude = larger.rounded_mean_bits(count as u64);
    debug_assert!(magnitude <= f64::MAX.to_bits());
    Ok(f64::from_bits(sign | magnitude))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimized_limb_division_matches_exact_integer_quotient_and_remainder() {
        let mut seed = 0x937a_f016_2c38_8b01_u64;
        for count in (1_u64..=64).chain([256, 1023, 8192, 16_384, 19_999, 20_000]) {
            for len in 0..=LIMBS {
                let mut words = [0; LIMBS];
                for word in &mut words[..len] {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    *word = seed;
                }
                let mut exact = Magnitude(words);
                let remainder = exact.divide(count);
                assert!(remainder < count);
                // The unique quotient and remainder must reconstruct the
                // original. Use reverse multiply/add instead of division.
                let mut carry = u128::from(remainder);
                for (word, quotient) in words.into_iter().zip(exact.0) {
                    carry += u128::from(quotient) * u128::from(count);
                    assert_eq!(carry as u64, word);
                    carry >>= 64;
                }
                assert_eq!(carry, 0);
            }
        }
    }
}
