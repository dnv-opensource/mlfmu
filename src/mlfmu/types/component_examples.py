from mlfmu.types.fmu_component import FmiVariableType, Variable


def create_fmu_signal_example() -> Variable:
    return Variable(
        name="dis_yx",
        type=FmiVariableType.REAL,
        description=None,
        start_value=None,
        is_array=None,
        length=None,
        variability=None,
    )
