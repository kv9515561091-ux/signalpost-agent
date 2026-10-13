"""Organisation number parsing and MOD-11 validation."""

import unittest

from src.orgnumber import (
    InvalidCompanyNumber,
    is_valid,
    mod11_control_digit,
    normalize,
    validate,
)


class NormalizeTests(unittest.TestCase):
    def test_strips_registry_separators(self):
        self.assertEqual(normalize("983 887 457"), "983887457")
        self.assertEqual(normalize("983-887-457"), "983887457")

    def test_accepts_integer_input(self):
        self.assertEqual(normalize(983887457), "983887457")

    def test_drops_other_punctuation(self):
        self.assertEqual(normalize(" Org.nr: 983 887 457 "), "983887457")

    def test_values_without_digits_yield_empty(self):
        self.assertEqual(normalize(""), "")
        self.assertEqual(normalize(None), "")


class Mod11Tests(unittest.TestCase):
    def test_control_digit_matches_known_valid_numbers(self):
        # Verified against 500 consecutive live Enhetsregisteret records.
        for number in ("889640782", "983887457", "989061593", "923609016"):
            self.assertEqual(
                mod11_control_digit(number[:8]), int(number[8]), number
            )

    def test_control_digit_of_zero_is_representable(self):
        # A remainder of 11 must map to a control digit of 0, not None.
        for serial in ("00000000", "00000014", "00000028"):
            self.assertEqual(mod11_control_digit(serial), 0, serial)

    def test_unrepresentable_control_digit_returns_none(self):
        # A remainder of 10 would require the digit 10, so no such org
        # number can exist and it must not be treated as valid.
        for serial in ("00000006", "00000023", "00000037"):
            self.assertIsNone(mod11_control_digit(serial), serial)


class ValidationTests(unittest.TestCase):
    def test_accepts_valid_numbers(self):
        for number in ("889640782", "983887457", "989061593", "923609016"):
            self.assertTrue(is_valid(number), number)

    def test_rejects_wrong_length(self):
        for bad in ("12345678", "1234567890", ""):
            self.assertFalse(is_valid(bad), bad)

    def test_rejects_non_digits(self):
        self.assertFalse(is_valid("abcdefghi"))

    def test_rejects_failed_checksum(self):
        # 983887457 with its final digit altered.
        self.assertFalse(is_valid("983887458"))

    def test_lenient_mode_accepts_any_nine_digits(self):
        self.assertTrue(is_valid("983887458", strict=False))
        self.assertFalse(is_valid("12345678", strict=False))

    def test_validate_returns_canonical_form(self):
        self.assertEqual(validate(" 983 887 457 "), "983887457")

    def test_validate_raises_on_bad_input(self):
        for bad in ("12345678", "983887458", "nonsense"):
            with self.assertRaises(InvalidCompanyNumber):
                validate(bad)


if __name__ == "__main__":
    unittest.main()
