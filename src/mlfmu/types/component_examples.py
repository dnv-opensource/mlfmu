def create_fmu_signal_example():
    """
    Create an example FMU signal variable.

    Returns
    -------
        Variable: An instance of the Variable class representing the FMU signal variable.
    """
    from mlfmu.types.fmu_component import FmiVariableType, Variable

    return Variable(
        name="dis_yx",
        type=FmiVariableType.REAL,
        description=None,
        start_value=None,
        is_array=None,
        length=None,
        variability=None,
    )
