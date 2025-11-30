# src/settings.py

class Settings:
    def __init__(self, mode="shielding", particles=1000, batches=10):
        self.mode = mode.lower()
        self.num_particles = particles
        self.batches = batches  # Relevant for K-eigenvalue later

        # Default defaults based on mode
        if self.mode == "shielding":
            self.use_implicit_capture = True
            self.weight_cutoff = 0.0001
            self.roulette_survival_prob = 0.1
        elif self.mode == "criticality":
            self.use_implicit_capture = False # Analog is safer for basic k-eff
            self.weight_cutoff = 0.0      # Not needed for analog
            self.roulette_survival_prob = 1.0
        else:
            raise ValueError(f"Unknown mode: {mode}")
