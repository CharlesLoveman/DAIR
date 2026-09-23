import numpy as np
import pandas as pd

# Environmental properties, those that affect the movement of the device and which could sensibly be known are listed as OBSERVABLE
# All are fixed for a single episode, which is a simplification 
env_config = {
    "flow_rate" : { "min" : 0.1, "max" : 5.0},                 # OBSERVABLE - device stops and measures current pipe flow rate off a spinner - measurement assumed perfect for now
    "temperature" : { "min" : 40, "max" : 120 },               # OBSERVABLE - this affects viscosity - measurement assumed perfect and LTE (local thermodynamic equilibrium so what the gauge reads applies to the pipe section
    "viscosity_0" : { "min" : 2.314e-5, "max" : 2.515e-5},     # NOT OBSERVABLE - property of the bulk flow, we assume it is +/-0.1e-5 Pa.s of water (1 Cp = 1mPa.s)
    "leak_flow_rate" : { "min" : 0.1, "max" : 5.0},            # NOT OBSERVABLE - a property of the leak function
    "leak_temperature" : { "min" : 40, "max" : 120 },          # NOT OBSERVABLE - a property of the leak function
    "sigma_left" : { "min" : 5, "max" : 20 },                  # NOT OBSERVABLE - only used if the leak plume is asymmetric
    "sigma_right" : { "min" : 20, "max" : 60 },                # NOT OBSERVABLE - only used if the leak plume is asymmetric
    "bias" : { "min" : -0.5, "max" : 0.5 },                    # NOT OBSERVABLE - sensor bias
    "drag_coefficient" : { "min" : 0.05, "max" : 0.40 },       # NOT OBSERVABLE - affects how much an impulse leads to a change in position
    "undulation_phase" : { "min" : 0.0, "max" : 3.142 },       # OBSERVABLE - proxy for wellbore trajectory information, only valid for undulating wells 
    "undulation_period" : { "min" : 2743.2, "max" : 10000.0 }, # OBSERVABLE - proxy for wellbore trajectory information, only valid for undulating wells (2743.2 m = 9000 ft which is 2-degrees per 100 ft 
    "incline" : { "min" : -1.0, "max" : 1.0 },                 # OBSERVABLE - proxy for wellbore trajectory information, only valid for inclined pipe
    "gravity_gain" : { "min" : 0.0, "max" : 0.05 },            # NOT OBSERVABLE - The extent to which gravity affects the impulse (>0 requires some compensation for pipe orientation)
    "actuator_gain" : { "min" : 0.9, "max" : 1.1 },            # NOT OBSERVABLE - The extent to which a request is over or under employed
    "actuator_lag" : { "min" : 0.75, "max" : 0.99 }            # NOT OBSERVABLE - The extent to which the proposed actuation overcomes the existing actuation 
    }

# Environment Class - consumes the env_config and handles the creation of the environment and the modelling of the robot (in the plant_model() method)
class Environment:
    def __init__(self, config, seed=1511):
        self.rng = np.random.default_rng(seed)
        self.env_config = config
        self.env_case = {}
        

    # initialise_episode_environment - samples one case and stores it in self.env_case
    def initialise_episode_environment(self):
        plume_type = self.rng.choice([ "gaussian", "lorentzian",
                                       "exponential", "asymmetric" ])

        # Pipe flow rate...
        flow_rate = self.rng.uniform(self.env_config["flow_rate"]["min"], self.env_config["flow_rate"]["max"])
        temperature = self.rng.uniform(self.env_config["temperature"]["min"], self.env_config["temperature"]["max"])
        # Leak flow rate...
        leak_flow_rate = self.rng.uniform(self.env_config["leak_flow_rate"]["min"], self.env_config["leak_flow_rate"]["max"])
        leak_temperature = self.rng.uniform(self.env_config["leak_temperature"]["min"], self.env_config["leak_temperature"]["max"])

        base_A = 100.0
        base_sigma = 15.0
        
        A = base_A * leak_flow_rate

        sigma = ( base_sigma * np.sqrt(leak_flow_rate) )
        sigma *= ( 1.0 + 0.005 * (leak_temperature - 20) )

        sigma_left = self.rng.uniform(self.env_config["sigma_left"]["min"], self.env_config["sigma_left"]["max"])
        sigma_right = self.rng.uniform(self.env_config["sigma_right"]["min"], self.env_config["sigma_right"]["max"])
        bias = self.rng.uniform(self.env_config["bias"]["min"],self.env_config["bias"]["max"])
        viscosity = self.rng.uniform(self.env_config["viscosity_0"]["min"], self.env_config["viscosity_0"]["max"]) * np.power(10,247.8/(temperature+133.15))*1000 # Vogel equation, good in the 10-100C range. 
        drag_coefficient = self.rng.uniform(self.env_config["drag_coefficient"]["min"],self.env_config["drag_coefficient"]["max"])

        pipe_profile_type = self.rng.choice(["flat","inclined","undulating"])

        undulation_phase = self.rng.uniform(self.env_config["undulation_phase"]["min"],self.env_config["undulation_phase"]["max"])
        undulation_period = self.rng.uniform(self.env_config["undulation_period"]["min"],self.env_config["undulation_period"]["max"])
        incline = self.rng.uniform(self.env_config["incline"]["min"],self.env_config["incline"]["max"])

        gravity_gain = self.rng.uniform(self.env_config["gravity_gain"]["min"],self.env_config["gravity_gain"]["max"])
        actuator_gain = self.rng.uniform(self.env_config["actuator_gain"]["min"],self.env_config["actuator_gain"]["max"])
        actuator_lag = self.rng.uniform(self.env_config["actuator_lag"]["min"],self.env_config["actuator_lag"]["max"])


        self.env_case = {
            "drift" : 0.0,
            "plume_type": plume_type,
            "leak_flow_rate": leak_flow_rate,
            "temperature": temperature,
            "A": A,
            "sigma": sigma,
            "sigma_left": sigma_left,
            "sigma_right": sigma_right,
            "bias": bias,

            "viscosity": viscosity,
            "drag_coefficient": drag_coefficient,
            "flow_rate" : flow_rate,

            "pipe_profile_type": pipe_profile_type,
            "undulation_phase": undulation_phase,
            "undulation_period": undulation_period,
            
            "gravity_gain": gravity_gain,
            "actuator_gain": actuator_gain,
            "actuator_lag": actuator_lag,
            "incline": incline
        }

    ## d_simulated (in training), used as d_clean for testing
    def concentration( self, x, x_star):
        pt = self.env_case["plume_type"]

        if pt == "gaussian":
            return self._plume_gaussian( x, x_star, self.env_case["A"], self.env_case["sigma"] )

        if pt == "lorentzian":
            return self._plume_lorentzian( x, x_star, self.env_case["A"], self.env_case["sigma"] )

        if pt == "exponential":
            return self._plume_exponential( x, x_star, self.env_case["A"], self.env_case["sigma"] )

        return self._asymmetric_gaussian( x, x_star, self.env_case["A"], self.env_case["sigma_left"], self.env_case["sigma_right"] )



    ## Different possible leak types with different signals
    def _plume_gaussian(self, x, x_star, A, sigma):
        return A * np.exp( -((x - x_star) ** 2) / (2 * sigma ** 2))

    def _plume_lorentzian(self, x, x_star, A, gamma):
        return A / ( 1.0 + ((x - x_star) / gamma) ** 2)

    def _plume_exponential(self, x, x_star, A, length):
        return A * np.exp( -abs(x - x_star) / length )

    def _asymmetric_gaussian(self, x, x_star, A, sigma_left, sigma_right):
        dx = x - x_star
        sigma = (sigma_left if dx < 0 else sigma_right)
        return A * np.exp( -(dx ** 2) / (2 * sigma ** 2))

    ## pipe slope for different possible sloping types
    def pipe_slope(self,x):
        profile = self.env_case["pipe_profile_type"]

        if profile == "flat":
            return 0.0

        if profile == "inclined":
            return self.env_case["incline"]

        return np.sin((self.env_case["undulation_phase"]) + (2.0 * np.pi * x / self.env_case["undulation_period"]))

    def sensor_drift(self):
        self.env_case["drift"] += self.rng.normal(0, 0.001) # sensor drift is a small random walk
        return self.env_case["drift"]

    ## physical plant model - THIS IS THE ROBOT UPDATING GIVEN THE u_cmd FROM THE CONTROLLER
    def plant_model( self, x, u_cmd, actuator_state):

        lag = self.env_case["actuator_lag"]

        actuator_state = ((1.0 - lag) * actuator_state + lag * u_cmd)
        slope = self.pipe_slope( x)
        gravity_term = ( self.env_case["gravity_gain"] * slope)
        drag_loss = ( self.env_case["drag_coefficient"] * self.env_case["flow_rate"] * actuator_state )
        viscous_loss = ( self.env_case["viscosity"] * 0.05 * np.sign(actuator_state) * min( abs(actuator_state), 1.0 ))
        u_actual = ( self.env_case["actuator_gain"] * actuator_state - drag_loss - gravity_term - viscous_loss )

        return ( u_actual, actuator_state, slope )



class GoalSeekingSimulator:

    def __init__(self, env, eta=0.05, delta=0.05, domain=(-100, 100), func=None):
        self.env = env
        self.eta = eta
        self.delta = delta
        self.domain = domain
        self.func = func # the function (under test or simulation)

    def sample_target(self):
        return self.env.rng.uniform(*self.domain)

    def sample_start(self):
        return self.env.rng.uniform(*self.domain)


    def d_clean( self, x, x_star):
        h = 0.5

        left = self.env.concentration( x - h, x_star )
        right = self.env.concentration( x + h, x_star )

        return (right - left) / (2 * h)

    # array based sensing - 5 values
    def get_sensor_values(self,x,x_star,lag_state,drift):
        offsets = [-2, -1, 0, 1, 2]
        outputs = []
        alpha = 0.1

        for i, off in enumerate(offsets):
            true_signal = self.env.concentration(x + off, x_star)
            lag_state[i] = ((1 - alpha)*lag_state[i] + alpha*true_signal)

            noisy = (lag_state[i] + self.env.env_case["bias"] + drift + self.env.rng.uniform(-self.delta, self.delta))
            outputs.append(noisy)

        return outputs, lag_state

    def d_observed( self, sensors ):
        if self.func is not None:
            return self.func(sensors)
        # default d(x) - a 
        return (sensors[3] - sensors[1]) / 2.0

    def controller(self, d_est):
        return self.eta * d_est


    # in an episode we start at a random location relative to a random target in a sampled environment, and then have
    # horizon number of steps to approach the target.
    
    def run_episode(self,episode_id,horizon):

        rows = []

        x_star = self.sample_target()

        x_real = self.sample_start()

        self.env.initialise_episode_environment()

        drift = 0.0
        lag_state = [0.0] * 5
        actuator_state = 0.0

        for step in range(horizon):

            drift = self.env.sensor_drift() # drift is a small random walk
            sensors, lag_state = ( self.get_sensor_values(x_real, x_star, lag_state, drift) )

            d_clean = self.d_clean(x_real,x_star) # ideal behaviour
            d_obs = self.d_observed(sensors) # this is the learned behaviour

            u_cmd = self.controller(d_obs) # output of the controller
            x_pred = x_real + u_cmd        # predicted location of the robot

            (u_actual,actuator_state,slope) = self.env.plant_model(x_real,u_cmd,actuator_state) # the actual effect in the environment
            x_real_next = (x_real + u_actual) # the real location of the robot

            

            rows.append({
                "episode_id": episode_id,
                "step": step,

                "x_real": x_real,
                "x_star": x_star,

                "error": x_real - x_star,

                "x_pred": x_pred,
                "x_real_next": x_real_next,
                "prediction_error": x_pred - x_real_next,

                "flow_rate": self.env.env_case["flow_rate"],
                "temperature": self.env.env_case["temperature"],
                "plume_type": self.env.env_case["plume_type"],
                "bias": self.env.env_case["bias"],
                "drift": drift,
                "viscosity": self.env.env_case["viscosity"],
                "drag_coefficient": self.env.env_case["drag_coefficient"],
                "pipe_profile_type": self.env.env_case["pipe_profile_type"],
                "undulation_phase": self.env.env_case["undulation_phase"],
                "undulation_period": self.env.env_case["undulation_period"],
                "gravity_gain": self.env.env_case["gravity_gain"],
                "actuator_gain": self.env.env_case["actuator_gain"],
                "actuator_lag": self.env.env_case["actuator_lag"],
                "pipe_slope": slope,

                "sensor_m2": sensors[0],
                "sensor_m1": sensors[1],
                "sensor_0": sensors[2],
                "sensor_p1": sensors[3],
                "sensor_p2": sensors[4],

                "m_val" : np.abs(d_clean)/(np.abs(x_real-x_star)),  #+1e-12), # m >= 0.0

                "d_clean": d_clean,
                "d_observed": d_obs,
                "u_cmd": u_cmd,
                "u_actual": u_actual
            })

            x_real = x_real_next # update the robot position for the next round

        return rows



def generate_dataset( output_csv, env, num_episodes=5000, horizon=250, func=None):

    sim = GoalSeekingSimulator(env,func=func)

    rows = []
    for ep in range(num_episodes):
        rows.extend(sim.run_episode( ep, horizon))

        if ep % 100 == 0:
            print(f"episode {ep}")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv,index=False)

    print(f"Saved {len(df)} rows")
    print(df.head())

    # Summary
    numeric_cols = df.select_dtypes(include=np.number).columns
    summary = pd.DataFrame({
        "min": df[numeric_cols].min(),
        "max": df[numeric_cols].max(),
        "mean": df[numeric_cols].mean(),
        "std": df[numeric_cols].std(),
        "p01": df[numeric_cols].quantile(0.01),
        "p50": df[numeric_cols].median(),
        "p99": df[numeric_cols].quantile(0.99),
    })

    print(summary.sort_index())


# CLASSICAL LOCAL GRADIENT CONTROLLERS
# simplest controller
def d_fd(sensors):
    return (sensors[3]-sensors[1])/2.0
# stencil-based controller
def d_fd_stencil(sensors):
    return (-2*sensors[0]-sensors[1]+sensors[3]+2*sensors[4])/6.0
# can a learned function d(x) produce stronger goal-seeking behaviour,
# larger certified contraction regions, and smaller guaranteed leak localisation
# error than these finite-difference baselines?
#
# can use temperature, flow-rate, pipe 
if __name__ == "__main__":
    env = Environment(env_config)
    
    generate_dataset(
        "goal_seeking_dataset.csv",env,
        num_episodes=10000,
        horizon=250
    )
    
