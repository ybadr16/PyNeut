# tests/mock_components.py
class MockReader:
    """A fake CrossSectionReader that returns fixed constants."""
    def __init__(self, sigma_s, sigma_a):
        self.sigma_s = sigma_s
        self.sigma_a = sigma_a
        self.sigma_t = sigma_s + sigma_a

    def get_cross_sections(self, element, energy, sampler, N):
        # Return: sigma_s, sigma_a, sigma_f, Sigma_t
        # We ignore 'element', 'energy', and 'sampler' for the test
        # We assume N=1.0 so microscopic = macroscopic
        return self.sigma_s, self.sigma_a, 0.0, self.sigma_t
