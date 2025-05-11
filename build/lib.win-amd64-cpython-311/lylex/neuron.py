"""
neuron.py - Wrapper around the NEURON simulation environment.
"""

from typing import Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from neuron import h


class Neuron:
    """
    Wrapper for NEURON HOC interface.
    """

    def __init__(self, hoc_file: str = None):
        if hoc_file:
            h.load_file(hoc_file)

    def exec(self, statement: str):
        """
        Execute a HOC statement.
        """
        return h(statement)

    def set_tstop(self, tstop: float):
        """
        Set simulation end time.
        """
        h.tstop = tstop

    def set_dt(self, dt: float):
        """
        Set simulation time step.
        """
        h.dt = dt

    def run(self):
        """
        Run the simulation.
        """
        h.run()

    def record_trace(
        self, ref=None, t_ref=None, run: bool = True, use_jax: bool = False
    ):
        """
        Record a variable over time.

        Parameters:
            ref: HOC reference (e.g., section(0.5)._ref_v).
            t_ref: time reference (defaults to h._ref_t).
            run: whether to run simulation before returning.
            use_jax: if True return JAX arrays

        Returns:
            t: numpy array of times.
            values: numpy array of recorded values.
        """
        t_ref = t_ref or h._ref_t
        ref = ref or h._ref_v
        t_vec = h.Vector()
        t_vec.record(t_ref)
        v_vec = h.Vector()
        v_vec.record(ref)
        if run:
            h.run()
        # Convert to arrays (NumPy or JAX)
        if use_jax:
            t = jnp.array(list(t_vec))
            v = jnp.array(list(v_vec))
        else:
            t = np.array(list(t_vec))
            v = np.array(list(v_vec))
        return t, v

    def record_multiple_traces(
        self, refs: dict, t_ref=None, run: bool = True, use_jax: bool = False
    ):
        """
        Record multiple variables over time.

        Parameters:
            refs: mapping of variable name -> HOC reference (e.g., {'v': section(0.5)._ref_v})
            t_ref: time reference (defaults to h._ref_t)
            run: whether to execute the simulation
            use_jax: if True return JAX arrays

        Returns:
            dict mapping name -> (t_array, values_array)
        """
        t_ref = t_ref or h._ref_t
        # Record time
        t_vec = h.Vector()
        t_vec.record(t_ref)
        # Record each variable
        vecs = {}
        for name, ref_item in refs.items():
            v_vec = h.Vector()
            v_vec.record(ref_item)
            vecs[name] = v_vec
        if run:
            h.run()
        # Convert to arrays
        t_list = list(t_vec)
        if use_jax:
            t_arr = jnp.array(t_list)
            traces = {name: (t_arr, jnp.array(list(vv))) for name, vv in vecs.items()}
        else:
            t_arr = np.array(t_list)
            traces = {name: (t_arr, np.array(list(vv))) for name, vv in vecs.items()}
        return traces

    def sweep_parameter(
        self,
        param_name: str,
        values: List,
        param_name2: Optional[str] = None,
        values2: Optional[List] = None,
        ref=None,
        t_ref=None,
        use_jax: bool = False,
    ):
        """
        Sweep one or two NEURON parameters and record a trace for each combination.
        If param_name2 and values2 are provided, a 2D sweep is performed.

        Returns:
            If 1D sweep: dict mapping {value: (t_array, v_array)}
            If 2D sweep: dict mapping {value1: {value2: (t_array, v_array)}}
        """
        # Ensure parameter exists
        if not hasattr(h, param_name):
            raise AttributeError(f"NEURON has no parameter '{param_name}'")
        original = getattr(h, param_name)
        results = {}

        original2 = None
        if param_name2:
            if not hasattr(h, param_name2):
                setattr(h, param_name, original)  # Restore first param before erroring
                raise AttributeError(f"NEURON has no parameter '{param_name2}'")
            original2 = getattr(h, param_name2)

        for val in values:
            setattr(h, param_name, val)
            if param_name2 and values2:
                current_sweep_results = {}
                for val2 in values2:
                    setattr(h, param_name2, val2)
                    t, v = self.record_trace(
                        ref=ref, t_ref=t_ref, run=True, use_jax=use_jax
                    )
                    current_sweep_results[val2] = (t, v)
                results[val] = current_sweep_results
            else:
                t, v = self.record_trace(
                    ref=ref, t_ref=t_ref, run=True, use_jax=use_jax
                )
                results[val] = (t, v)

        # restore original parameter(s)
        setattr(h, param_name, original)
        if param_name2 and original2 is not None:
            setattr(h, param_name2, original2)
        return results

    def export_trace(self, t, v, file_path: str, metadata: Optional[dict] = None):
        """
        Export a single trace to disk as a .npz file.
        Optional metadata can be included in the .npz file.
        """
        to_save = {"t": t, "v": v}
        if metadata:
            to_save.update(metadata)
        np.savez(file_path, **to_save)
        # export complete (no return)

    def plot_trace(
        self,
        traces: Union[
            Tuple[np.ndarray, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]]
        ],
        title: Optional[str] = None,
        xlabel: str = "Time",
        ylabel: str = "Value",
        save_path: Optional[str] = None,
    ):
        """
        Plot one or more NEURON traces using matplotlib.

        Parameters:
            traces: Either a single (t, v) tuple or a dict {'label': (t,v)} for multiple traces.
            title: Optional title for the plot.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            save_path: Optional path to save the figure.
        """
        import matplotlib.pyplot as plt

        plt.figure()

        if isinstance(traces, tuple) and len(traces) == 2:
            t, v = traces
            plt.plot(t, v)
        elif isinstance(traces, dict):
            for label, (t, v) in traces.items():
                plt.plot(t, v, label=label)
            if len(traces) > 1:
                plt.legend()
        else:
            raise ValueError(
                "traces must be a (t,v) tuple or a dict of {'label': (t,v)}"
            )

        if title:
            plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if save_path:
            plt.savefig(save_path)
        plt.show()
