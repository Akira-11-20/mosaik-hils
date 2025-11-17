"""
Inverse Compensator Simulator
Applies compensation gain to current position
"""
import mosaik_api_v3 as mosaik_api


class InverseCompensator:
    """
    Inverse compensator that uses previous position to predict and compensate
    This modulates the feedback signal going to the PD controller
    """

    def __init__(self, comp_gain=1.0, dt=0.01):
        """
        Args:
            comp_gain: Compensation gain to apply (higher = more aggressive prediction)
            dt: Time step [s]
        """
        self.comp_gain = comp_gain
        self.dt = dt
        self.output = 0.0
        self.prev_position = 0.0

    def step(self, current_position):
        """
        Apply inverse compensation using previous position

        Uses the difference between current and previous position to predict
        future position and compensate accordingly.

        Compensation formula:
        position_change = current_position - prev_position
        predicted_position = current_position + comp_gain * position_change

        Args:
            current_position: Current position from PD controller
        Returns:
            compensated_position: Position with compensation applied
        """
        # Apply inverse compensation formula (same as HILS/Orbital)
        # Formula: y_comp[k] = gain * y[k] - (gain - 1) * y[k-1]
        #
        # This can be rewritten as: y_comp[k] = y[k] + (gain - 1) * (y[k] - y[k-1])
        #
        # This is a lead compensator that predicts future position based on
        # the current trend (position change).
        #
        # comp_gain > 1.0: aggressive prediction (leads further)
        # comp_gain = 1.0: no compensation (pass-through)
        # comp_gain < 1.0: damping (conservative)
        compensated_position = self.comp_gain * current_position - (self.comp_gain - 1.0) * self.prev_position

        self.output = compensated_position

        # Update for next step
        self.prev_position = current_position

        return self.output


class InverseCompensatorSimulator(mosaik_api.Simulator):
    """Mosaik simulator for inverse compensator"""

    def __init__(self):
        super().__init__({
            'models': {
                'InverseCompensator': {
                    'public': True,
                    'params': ['comp_gain', 'dt'],
                    'attrs': ['position', 'output'],
                },
            },
        })
        self.sid = None
        self.time_resolution = None
        self.entities = {}
        self.eid_prefix = 'InverseComp_'

    def init(self, sid, time_resolution=1.0, **sim_params):
        self.sid = sid
        self.time_resolution = time_resolution
        return self.meta

    def create(self, num, model, **model_params):
        entities = []

        for _ in range(num):
            eid = f'{self.eid_prefix}{len(self.entities)}'

            # Create compensator instance
            compensator = InverseCompensator(
                comp_gain=model_params.get('comp_gain', 1.0),
                dt=model_params.get('dt', 0.01)
            )

            self.entities[eid] = compensator
            entities.append({'eid': eid, 'type': model})

        return entities

    def step(self, time, inputs, max_advance):
        """Execute simulation step"""
        for eid, entity_inputs in inputs.items():
            compensator = self.entities[eid]

            # Get current position
            position = 0.0
            if 'position' in entity_inputs:
                values = list(entity_inputs['position'].values())
                if values:
                    position = values[0]
            self.position = 10-position
            # Apply compensation
            compensator.step(10-position)

        return int(time + self.time_resolution)

    def get_data(self, outputs):
        """Return requested output data"""
        data = {}

        for eid, attrs in outputs.items():
            data[eid] = {}
            compensator = self.entities[eid]

            for attr in attrs:
                if attr == 'output':
                    data[eid][attr] = compensator.output
                elif attr == 'position':
                    data[eid][attr] = self.position

        return data


if __name__ == '__main__':
    mosaik_api.start_simulation(InverseCompensatorSimulator())
