from pydantic import BaseModel


class HousePrice(BaseModel):
    bed: str
    bathroom: str
    year_built: str
    heating: str
    Property_type: str
    area: float
    county: str
    zipcode: str
