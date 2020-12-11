from dateutil import parser

class HistoricTrendDTO():
    def __init__(self, start_date, end_date, measuring_point_id, level, measuring_point_ids=None):
        # self.StartDate = parser.parse(start_date)
        # self.EndDate = parser.parse(end_date)
        self.StartDate = start_date
        self.EndDate = end_date
        self.MeasuringPointID = measuring_point_id
        self.MeasuringPointIDList = measuring_point_ids
        self.Level = level

    StartDate = None
    EndDate = None
    OrganizationID = None
    MeasuringPointType = None
    MeasuringPointID = None
    MeasuringPointIDList = None
    AssemblyLineID = None
    Level = None
