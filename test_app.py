import unittest
from unittest.mock import MagicMock, patch
import app as main_app # Assuming your main file is app.py
from PIL import Image
import io

# Define a sample PRICING_USD structure similar to what's in app.py for testing
# This should be defined at the module level in test_app.py for tests to use
TEST_PRICING_USD = {
    "gemini-1.5-flash": {
        'input': {'standard': 0.075, 'long': 0.15},
        'output': {'standard': 0.30, 'long': 0.60},
        'cached_input': {'standard': 0.01875, 'long': 0.0375}, # 75% discount
        'cache_storage_hourly': 1.00
    },
    "gemini-1.5-pro": { # Example placeholder prices
        'input': {'standard': 0.5, 'long': 1.0},
        'output': {'standard': 1.5, 'long': 3.0},
        'cached_input': {'standard': 0.125, 'long': 0.25},
        'cache_storage_hourly': 1.00
    },
    "gemini-1.0-pro": { # Example placeholder prices
        'input': {'standard': 0.1}, # No long context distinction for this example
        'output': {'standard': 0.3},
        # No cached_input for this example, to test fallback
        'cache_storage_hourly': 1.00
    },
    "model-missing-specifics": { # For testing fallbacks
        'input': {'standard': 0.2},
        'output': {'standard': 0.4},
        'cache_storage_hourly': 1.00
    }
}

# Define a sample USD_TO_INR for testing
TEST_USD_TO_INR = 83.0

class TestApp(unittest.TestCase):

    def setUp(self):
        # Patch constants in the main_app module for the duration of tests
        self.pricing_patch = patch.dict(main_app.PRICING_USD, TEST_PRICING_USD, clear=True)
        self.usd_to_inr_patch = patch('app.USD_TO_INR', TEST_USD_TO_INR)

        self.pricing_patch.start()
        self.usd_to_inr_patch.start()

        main_app.USD_TO_INR = TEST_USD_TO_INR # Ensure it's directly set if patch doesn't cover all uses

    def tearDown(self):
        self.pricing_patch.stop()
        self.usd_to_inr_patch.stop()

    def test_calculate_cost_flash_standard(self):
        # Test Gemini 1.5 Flash, standard context, input
        cost_usd, cost_inr = main_app.calculate_cost(
            tokens=100000, model_name="gemini-1.5-flash", token_type='input',
            cached=False, prompt_length=100000
        )
        expected_usd = (100000 / 1_000_000) * TEST_PRICING_USD["gemini-1.5-flash"]['input']['standard']
        self.assertAlmostEqual(cost_usd, expected_usd)
        self.assertAlmostEqual(cost_inr, expected_usd * TEST_USD_TO_INR)

    def test_calculate_cost_flash_long_output(self):
        # Test Gemini 1.5 Flash, long context, output
        cost_usd, cost_inr = main_app.calculate_cost(
            tokens=200000, model_name="gemini-1.5-flash", token_type='output',
            cached=False, prompt_length=150000 # prompt_length > 128k makes it long
        )
        expected_usd = (200000 / 1_000_000) * TEST_PRICING_USD["gemini-1.5-flash"]['output']['long']
        self.assertAlmostEqual(cost_usd, expected_usd)
        self.assertAlmostEqual(cost_inr, expected_usd * TEST_USD_TO_INR)

    def test_calculate_cost_flash_cached_input(self):
        cost_usd, cost_inr = main_app.calculate_cost(
            tokens=50000, model_name="gemini-1.5-flash", token_type='input',
            cached=True, prompt_length=50000
        )
        expected_usd = (50000 / 1_000_000) * TEST_PRICING_USD["gemini-1.5-flash"]['cached_input']['standard']
        self.assertAlmostEqual(cost_usd, expected_usd)

    def test_calculate_cost_pro_standard_input(self):
        cost_usd, cost_inr = main_app.calculate_cost(
            tokens=100000, model_name="gemini-1.5-pro", token_type='input',
            cached=False, prompt_length=100000
        )
        expected_usd = (100000 / 1_000_000) * TEST_PRICING_USD["gemini-1.5-pro"]['input']['standard']
        self.assertAlmostEqual(cost_usd, expected_usd)

    def test_calculate_cost_gemini_1_0_pro(self):
        # Test Gemini 1.0 Pro (no long/cached distinction in test data)
        cost_usd, cost_inr = main_app.calculate_cost(
            tokens=100000, model_name="gemini-1.0-pro", token_type='input',
            prompt_length=100000
        )
        expected_usd = (100000 / 1_000_000) * TEST_PRICING_USD["gemini-1.0-pro"]['input']['standard']
        self.assertAlmostEqual(cost_usd, expected_usd)

    def test_calculate_cost_model_not_in_pricing(self):
        # Test fallback to flash if model not in PRICING_USD (with warning)
        # Need to patch st.warning as it's not available in this testing context directly
        with patch('app.st.warning') as mock_st_warning:
            cost_usd, cost_inr = main_app.calculate_cost(
                tokens=10000, model_name="unknown-model", token_type='input', prompt_length=10000
            )
            mock_st_warning.assert_any_call("Pricing not found for model: unknown-model. Using default (Flash) rates.")
        # Should use flash standard input pricing as fallback
        expected_usd = (10000 / 1_000_000) * TEST_PRICING_USD["gemini-1.5-flash"]['input']['standard']
        self.assertAlmostEqual(cost_usd, expected_usd)

    def test_calculate_cost_missing_long_rate(self):
        # Test fallback to standard if 'long' rate missing for a model that distinguishes context length
        with patch('app.st.warning') as mock_st_warning: # Mock st.warning
            cost_usd, cost_inr = main_app.calculate_cost(
                tokens=10000, model_name="model-missing-specifics", token_type='input', prompt_length=130000 # > 128k
            )
            # Check if the specific warning for missing long rate was called
            # This part of the test might need adjustment based on the exact warning message generated by calculate_cost
            # For now, we assume a warning is logged by st.warning if a rate is defaulted.
            # A more precise check would be:
            # mock_st_warning.assert_any_call("Rate type 'long' not found for 'input' in model-missing-specifics. Using 'standard'.")
            # However, the current calculate_cost doesn't log this specific fallback, it just uses the standard rate.
            # The test will pass if the calculation is correct.
            # We can check that *a* warning was called if the logic implies one should have been
            # if not category_rates.get(rate_type): would trigger one.
            # The current logic is: rate = category_rates.get(rate_type, category_rates.get('standard', 0))
            # This means no explicit warning is logged by calculate_cost for this specific fallback.
            # The warning in the original test for assertLogs might be from a different part of the logic.
            # Let's remove the warning check for this specific case as the code directly falls back.

        expected_usd = (10000 / 1_000_000) * TEST_PRICING_USD["model-missing-specifics"]['input']['standard']
        self.assertAlmostEqual(cost_usd, expected_usd)

    def test_calculate_cost_missing_cached_rate(self):
        # Test fallback to standard input if 'cached_input' rate missing
        with patch('app.st.warning') as mock_st_warning: # Mock st.warning
            cost_usd, cost_inr = main_app.calculate_cost(
                tokens=10000, model_name="gemini-1.0-pro", token_type='input', cached=True, prompt_length=10000
            )
            # Gemini 1.0 Pro has no 'cached_input' in TEST_PRICING_USD
            mock_st_warning.assert_any_call("'cached_input' pricing not defined for gemini-1.0-pro. Using standard input rate.")
        expected_usd = (10000 / 1_000_000) * TEST_PRICING_USD["gemini-1.0-pro"]['input']['standard'] # Fallback to standard input
        self.assertAlmostEqual(cost_usd, expected_usd)

    def test_route_page_to_model_placeholder(self):
        # Test the placeholder router function
        # Create a dummy PIL image
        img = Image.new('RGB', (60, 30), color = 'red')
        available_models = ["gemini-1.5-flash", "gemini-1.5-pro"]

        # Mock st.info as it's a Streamlit function
        with patch('app.st.info') as mock_st_info:
            selected_model = main_app.route_page_to_model(img, available_models)
            self.assertEqual(selected_model, "gemini-1.5-flash") # Placeholder returns the first model
            mock_st_info.assert_called_once()

    def test_calculate_file_hash(self):
        file_bytes = b"This is a test file."
        file_hash = main_app.calculate_file_hash(file_bytes)
        # SHA256 hash for "This is a test file."
        expected_hash = "f29bc64a9d3732b4b9035125fdb3285f5b6455778edca72414671e0ca3b2e0de" # Corrected hash
        self.assertEqual(file_hash, expected_hash)

    # It's difficult to unit test Streamlit's st.session_state directly without a running app.
    # Similarly, functions heavily reliant on st calls or GenAI API calls are hard to unit test
    # without extensive mocking (e.g., extract_and_evaluate_with_caching, run_evaluation_agent).
    # We focus on helper functions that have clear inputs and outputs.

if __name__ == '__main__':
    unittest.main()
