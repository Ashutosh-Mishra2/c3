from c3po.signals.envelopes import *
from c3po.signals.component import Signal_component as Comp
from c3po.signals.signal import Signal as Signal

from c3po.signals.generator import Device as Device
from c3po.signals.generator import AWG as AWG
from c3po.signals.generator import Mixer as Mixer
from c3po.signals.generator import Generator as Generator

import uuid
import matplotlib.pyplot as plt


comp_group = uuid.uuid4()
carrier_group = uuid.uuid4()


flattop_params1 = {
    'amp' : 15e6 * 2 * np.pi,
    'T_up' : 5e-9,
    'T_down' : 45e-9,
    'xy_angle' : 0,
    'freq_offset' : 0e6 * 2 * np.pi
}

flattop_params2 = {
    'amp' : 3e6 * 2 * np.pi,
    'T_up' : 25e-9,
    'T_down' : 30e-9,
    'xy_angle' : np.pi / 2.0,
    'freq_offset' : 0e6 * 2 * np.pi
}

params_bounds = {
    'T_up' : [2e-9, 98e-9],
    'T_down' : [2e-9, 98e-9],
    'freq_offset' : [-1e9 * 2 * np.pi, 1e9 * 2 * np.pi]
}


def my_flattop(t, params):
    t_up = params['T_up']
    t_down = params['T_down']
    return flattop(t, t_up, t_down)


p1 = Comp(
    name = "pulse1",
    desc = "flattop comp 1 of signal 1",
    shape = my_flattop,
    params = flattop_params1,
    bounds = params_bounds,
    groups = [comp_group]
)
print("p1 uuid: " + str(p1.get_uuid()))

p2 = Comp(
    name = "pulse2",
    desc = "flattop comp 2 of signal 1",
    shape = my_flattop,
    params = flattop_params2,
    bounds = params_bounds,
    groups = [comp_group]
)
print("p2 uuid: " + str(p2.get_uuid()))

####
# Below code: For checking the single signal components
####

# t = np.linspace(0, 150e-9, int(150e-9*1e9))
# plt.plot(t, p1.get_shape_values(t))
# plt.plot(t, p2.get_shape_values(t))
# plt.show()


carrier_parameters = {
    'freq' : 6e9 * 2 * np.pi
}

carr = Comp(
    name = "carrier",
    desc = "Frequency of the local oscillator",
    params = carrier_parameters,
    groups = [carrier_group]
)
print("carr uuid: " + str(carr.get_uuid()))


comps = []
comps.append(carr)
comps.append(p1)
comps.append(p2)



sig = Signal()
sig.name = "signal1"
sig.t_start = 0
sig.t_end = 150e-9
sig.comps = comps


# print(sig.get_parameters())
# print(" ")
# print(" ")
# print(" ")

# print(sig.get_history())
# print(" ")
# print(" ")
# print(" ")


# sig.save_params_to_history("initial")

# print(sig.get_history())
# print(" ")
# print(" ")
# print(" ")


# sig.save_params_to_history("test2")

# print(sig.get_history())




class SignalSetup(Generator):

    def __init__(
            self,
            resolutions = {},
            groups = {}
            ):

        self.resolutions = resolutions
        self.groups = groups

        self.output = None


    def generate_signal(self, ressources):

        output = {}

        awg = self.devices["awg"]
        mixer = self.devices["mixer"]


        for sig in ressources:

            awg.t_start = sig.t_start
            awg.t_end = sig.t_end
            awg.resolutions = self.resolutions
            awg.ressources = [sig]
            awg.ressource_groups = self.groups
            awg.create_IQ("awg")

#            awg.plot_IQ_components("awg")


            mixer.t_start = sig.t_start
            mixer.t_end = sig.t_end
            mixer.resolutions = self.resolutions
            mixer.ressources = [sig]
            mixer.ressource_groups = self.groups
            mixer.calc_slice_num("sim")
            mixer.create_ts("sim")

            I = np.interp(mixer.ts, awg.ts, awg.get_I())
            Q = np.interp(mixer.ts, awg.ts, awg.get_Q())

            mixer.Inphase = I
            mixer.Quadrature = Q
            mixer.combine("sim")

            output[(sig.name,sig.get_uuid())] = {"ts" : mixer.ts}
            output[(sig.name,sig.get_uuid())].update({"signal" : mixer.output})

            self.output = output

        return output



    def plot_signals(self):

        for entry in self.output:
            signal_name = entry[0]
            signal = self.output[entry]

            """ Plotting control functions """
            plt.rcParams['figure.dpi'] = 100


            ts = signal["ts"]
            values = signal["signal"]

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(ts, values)
            ax.set_xlabel('Time [ns]')
            plt.title(signal_name)

            plt.show(block=False)


#     def plot_fft_signal(self):

        # print("WARNING: still have to adjust the x-axis")

        # """ Plotting control functions """
        # plt.rcParams['figure.dpi'] = 100
        # signal = self.generate_signal()

        # fft_signal = np.fft.fft(signal)
        # fft_signal = np.fft.fftshift(fft_signal.real / max(fft_signal.real))

        # plt.plot(self.ts[1] * self.res[0], fft_signal)

        # plt.show(block=False)
        # plt.show()



awg = AWG()
mixer = Mixer()

# print(len(awg.ts))
# print(len(awg.get_I()))


# plt.plot(awg.ts, awg.get_I())
# plt.plot(awg.ts, awg.get_Q())
# plt.show()


resolutions = {
    "awg" : 1e9,
    "sim" : 1e12
}

devices = {
    "awg" : awg,
    "mixer" : mixer
}


groups = {
    "comp" : comp_group,
    "carrier" : carrier_group
}


gen = SignalSetup()
gen.devices = devices
gen.resolutions = resolutions
gen.groups = groups


output = gen.generate_signal([sig])

gen.plot_signals()

# print(output)


# ts = output[(sig.name, sig.get_uuid())]["ts"]
# values = output[(sig.name, sig.get_uuid())]["signal"]


# plt.plot(ts, values)
# plt.show()
