"""
PD Controller Simulator
Simple PD controller for position tracking
"""

import mosaik_api_v3 as mosaik_api


class PDController:
    """PD Controller for position tracking with integrated position state"""

    def __init__(self, kp=1.0, kd=0.1, dt=0.01, initial_position=0.0):
        """
        Args:
            kp: Proportional gain
            kd: Derivative gain
            dt: Time step [s]
            initial_position: Initial position [m]
        """
        self.kp = kp
        self.kd = kd
        self.dt = dt

        # State variables
        self.position = initial_position
        self.velocity = 0.0
        self.prev_error = 0.0
        self.control_output = 0.0

    def step(self, target_position):
        """
        PD control law: u = Kp * e + Kd * de/dt
        Then integrate control output to get new position

        Args:
            target_position: Target position from plant
        Returns:
            current_position: Updated position after control
        """
        # Position error
        error = target_position - self.position

        # Derivative term (numerical derivative of error)
        derivative = (error - self.prev_error) / self.dt

        # PD control law (output is velocity command)
        self.control_output = self.kp * error + self.kd * derivative

        # Integrate to get new position
        self.velocity = self.control_output
        self.position += self.velocity * self.dt

        # Update for next step
        self.prev_error = error

        return self.position

    def get_error(self):
        """Return current error"""
        return self.prev_error


class PDControllerSimulator(mosaik_api.Simulator):
    """Mosaik simulator for PD controller"""

    def __init__(self):
        super().__init__(
            {
                "models": {
                    "PDController": {
                        "public": True,
                        "params": ["kp", "kd", "dt", "initial_position"],
                        "attrs": ["target_position", "position", "control_output", "velocity", "error"],
                    },
                },
            }
        )
        self.sid = None
        self.time_resolution = None
        self.entities = {}
        self.eid_prefix = "PDController_"

    def init(self, sid, time_resolution=1.0, **sim_params):
        self.sid = sid
        self.time_resolution = time_resolution
        return self.meta

    def create(self, num, model, **model_params):
        entities = []

        for _ in range(num):
            eid = f"{self.eid_prefix}{len(self.entities)}"

            # Create controller instance
            controller = PDController(
                kp=model_params.get("kp", 1.0),
                kd=model_params.get("kd", 0.1),
                dt=model_params.get("dt", 0.01),
                initial_position=model_params.get("initial_position", 0.0),
            )

            self.entities[eid] = controller
            entities.append({"eid": eid, "type": model})

        return entities

    def step(self, time, inputs, max_advance):
        """Execute simulation step"""
        for eid, entity_inputs in inputs.items():
            controller = self.entities[eid]

            # Get target position from plant (default to 10.0)
            target_position = 10.0
            if "target_position" in entity_inputs:
                target_values = list(entity_inputs["target_position"].values())
                if target_values:
                    target_position = target_values[0]

            # Execute PD control step
            controller.step(target_position)

        return int(time + self.time_resolution)

    def get_data(self, outputs):
        """Return requested output data"""
        data = {}

        for eid, attrs in outputs.items():
            data[eid] = {}
            controller = self.entities[eid]

            for attr in attrs:
                if attr == "position":
                    data[eid][attr] = controller.position
                elif attr == "control_output":
                    data[eid][attr] = controller.control_output
                elif attr == "velocity":
                    data[eid][attr] = controller.velocity
                elif attr == "error":
                    data[eid][attr] = controller.get_error()
                elif attr == "target_position":
                    # Echo back for debugging
                    data[eid][attr] = 0.0

        return data


if __name__ == "__main__":
    mosaik_api.start_simulation(PDControllerSimulator())
