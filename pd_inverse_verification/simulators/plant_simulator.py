"""
Simple 1D Plant Simulator
Outputs a constant target value
"""

import mosaik_api_v3 as mosaik_api


class SimplePlant:
    """Simple plant - outputs constant target value"""

    def __init__(self, target=10.0):
        """
        Args:
            target: Target value to output
        """
        self.target = target

    def step(self):
        """
        Output constant target value

        Returns:
            target: Constant target value
        """
        return self.target


class PlantSimulator(mosaik_api.Simulator):
    """Mosaik simulator for simple plant"""

    def __init__(self):
        super().__init__(
            {
                "models": {
                    "SimplePlant": {
                        "public": True,
                        "params": ["target"],
                        "attrs": ["target_output"],
                    },
                },
            }
        )
        self.sid = None
        self.time_resolution = None
        self.entities = {}
        self.eid_prefix = "Plant_"

    def init(self, sid, time_resolution=1.0, **sim_params):
        self.sid = sid
        self.time_resolution = time_resolution
        return self.meta

    def create(self, num, model, **model_params):
        entities = []

        for _ in range(num):
            eid = f"{self.eid_prefix}{len(self.entities)}"

            # Create plant instance with target value
            plant = SimplePlant(target=model_params.get("target", 10.0))

            self.entities[eid] = plant
            entities.append({"eid": eid, "type": model})

        return entities

    def step(self, time, inputs, max_advance):
        """Execute simulation step"""
        # Plant just outputs constant target value, no inputs needed
        for eid in self.entities:
            plant = self.entities[eid]
            plant.step()

        return int(time + self.time_resolution)

    def get_data(self, outputs):
        """Return requested output data"""
        data = {}

        for eid, attrs in outputs.items():
            data[eid] = {}
            plant = self.entities[eid]

            for attr in attrs:
                if attr == "target_output":
                    data[eid][attr] = plant.target

        return data


if __name__ == "__main__":
    mosaik_api.start_simulation(PlantSimulator())
