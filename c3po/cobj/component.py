import uuid

# this is originally the class "Pulse"
class Component:

    def __init__(
            self,
            name = " ",
            desc = " ",
            comment = " ",
            ):

        # make a random UUID which uniquely identifies/represents the component
        # https://docs.python.org/2/library/uuid.html#uuid.uuid4
        self.__uuid = uuid.uuid4()
        self.name = name
        self.desc = desc
        self.comment = comment

    def get_uuid(self):
        return self.__uuid

    def set_uuid(self, uuid):
        self.__uuid = uuid

class ControlComponent(Component):
    """Represents a pulse.

    Parameters
    ----------
    shape : func
        Function handle to function specifying the exact shape of the pulse
    parameters : dict
        dictionary of the parameters needed for the shape-function to
        create the desired pulse
    bounds : dict
        boundaries of the parameters, i.e. technical limits of experimental
        setup or physical boundaries. needed for optimizer
    """

    def __init__(
            self,
            name = " ",
            desc = " ",
            comment = " ",
            params = {},
            bounds = {},
            groups = [],
            shape = None
            ):
        super().__init__(
            name = name,
            desc = desc,
            comment = comment
            )
        self.params = params
        self.bounds = bounds
        self.groups = groups
        self.shape = shape

        # make a random UUID which uniquely identifies/represents the component
        # https://docs.python.org/2/library/uuid.html#uuid.uuid4
        self.__uuid = uuid.uuid4()

    def get_shape_values(self, ts):
        return self.shape(ts, self.params)

class PhysicalComponent(Component):
    def __init__(
            self,
            name = " ",
            desc = " ",
            comment = " ",
            hilbert_dim = None
            ):
        super().__init__(
            name = name,
            desc = desc,
            comment = comment
            )
        self.hilbert_dim = hilbert_dim
        self.values = {}

    def get_values(self):
        return self.values


class Qubit(PhysicalComponent):
    def __init__(
            self,
            name = " ",
            desc = " ",
            comment = " ",
            hilbert_dim = None,
            freq = None,
            delta = None,
            T1 = None,
            T2star = None,
            temp = None
            ):
        super().__init__(
            name = name,
            desc = desc,
            comment = comment,
            hilbert_dim = hilbert_dim
            )
        self.values['freq'] = freq
        if hilbert_dim > 2:
            self.values['delta'] = delta
        self.values['T1'] = T1
        self.values['T2star'] = T2star
        self.values['temp'] = temp

class Resonator(PhysicalComponent):
    def __init__(
            self,
            name = " ",
            desc = " ",
            comment = " ",
            hilbert_dim = None,
            freq = None
            ):
        super().__init__(
            name = name,
            desc = desc,
            comment = comment,
            hilbert_dim = hilbert_dim
            )
        self.values['freq'] = freq

class Coupling(PhysicalComponent):
    def __init__(
            self,
            name = " ",
            desc = " ",
            comment = " ",
            hilbert_dim = None,
            connected = None,
            strength = None
            ):
        super().__init__(
            name = name,
            desc = desc,
            comment = comment,
            hilbert_dim = hilbert_dim
            )
        self.values['strength'] = strength
        self.connected = connected

class Drive(PhysicalComponent):
    def __init__(
            self,
            name = " ",
            desc = " ",
            comment = " ",
            hilbert_dim = None,
            connected = None,
            strength = None
            ):
        super().__init__(
            name = name,
            desc = desc,
            comment = comment,
            hilbert_dim = hilbert_dim
            )
        self.connected = connected
