from raynx.models import InputModel, OutputModel


class TestInput1(InputModel):
    val: int
    offset: int = 0


class TestInput2(InputModel):
    val: str


class TestOutput1(OutputModel):
    val: float


class TestOutput2(OutputModel):
    val: str


class TOIter1(OutputModel):
    val: list[float]


class TIIter2(InputModel):
    val: list[float]
