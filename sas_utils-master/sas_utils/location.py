import pdb, math, numbers
import numpy as np

from shapely.geometry import Point

class Location(object):
  # Generic Location class designed to make keeping track of X and Y and Lat and Lon easier
  def __init__(self, ylat, xlon):
    if not isinstance(ylat, numbers.Number):
      raise ValueError("Invalid value for ylat: Received %s Expected Number" % str(type(ylat)))
    if not isinstance(xlon, numbers.Number):
      raise ValueError("Invalid value for xlon: Received %s Expected Number" % str(type(xlon)))
    self.lon = xlon
    self.lat = ylat
    self.x = xlon
    self.y = ylat

  def __str__(self):
    return "xLon: %.5f, yLat: %.5f" % (self.lon, self.lat)

  def __iter__(self):
    yield self.x
    yield self.y
    # return (self.x, self.y)

  def __eq__(self, other):
    if isinstance(other, Location):
      return (abs(self.lat - other.lat) < 1e-6) and (abs(self.lon - other.lon) < 1e-6)
    else:
      return False

  def __ne__(self, other):
    return not self.__eq__(other)

  def __add__(self, other):
    if isinstance(other, LocDelta):
      return Location(self.lat + other.d_ylat, self.lon + other.d_xlon)

  def __sub__(self, other):
    if isinstance(other, LocDelta):
      return Location(self.lat - other.d_ylat, self.lon - other.d_xlon)
    elif isinstance(other, Location):
      return LocDelta(self.lat - other.lat, self.lon - other.lon)

  def npArray(self):
    return np.array([self.lon, self.lat])

  def shapelyPoint(self):
    return Point([self.lon, self.lat])

  def asIntTuple(self):
    return (int(self.x), int(self.y))

  def asTuple(self):
    return (self.x, self.y)


class StampedLocation(object):
  """docstring for StampedLocation"""
  def __init__(self, loc, time):
    self.loc = loc
    self.time = time
    



class LocDelta(object):

  def __init__(self, d_ylat, d_xlon):
    if not isinstance(d_ylat, numbers.Number):
      raise ValueError("Invalid value for d_ylat: Received %s Expected Number" % str(type(d_ylat)))
    if not isinstance(d_xlon, numbers.Number):
      raise ValueError("Invalid value for d_xlon: Received %s Expected Number" % str(type(d_xlon)))

    self.d_ylat = d_ylat
    self.d_xlon = d_xlon

  @classmethod
  def fromPolar(cls, r, theta, angle_units="radians"):

    if (angle_units.lower() == "radians") or (angle_units.lower() == "rad"):
      dx = r*math.cos(theta)
      dy = r*math.sin(theta)
    elif (angle_units.lower() == "degrees") or (angle_units.lower() == "deg"): 
      dx = r*math.cos(math.radians(theta))
      dy = r*math.sin(math.radians(theta))
    return cls(d_ylat=dy, d_xlon=dx)

  def __str__(self):
    return "Delta xLon: %.5f, Delta yLat: %.5f" % (self.d_xlon, self.d_ylat)

  def __add__(self, other):
    if isinstance(other, Location):
      return Location(self.d_ylat + other.lat, self.d_xlon + other.lon)
    elif isinstance(other, LocDelta):
      return LocDelta(self.d_ylat + other.d_ylat, self.d_xlon + other.d_xlon)

  def __sub__(self, other):
    if isinstance(other, Location):
      return Location(self.d_ylat - other.lat, self.d_xlon - other.lon)
    elif isinstance(other, LocDelta):
      return LocDelta(self.d_ylat - other.d_ylat, self.d_xlon - other.d_xlon)

  def __mul__(self, other):
    if isinstance(other, float) or isinstance(other, int):
      return LocDelta(self.d_ylat*float(other),self.d_xlon*float(other))

  def __rmul__(self, other):
    return self.__mul__(other)

  def __div__(self, other):
    if isinstance(other, float) or isinstance(other, int):
      return LocDelta(self.d_ylat/float(other),self.d_xlon/float(other))

  def __eq__(self, other):
    if isinstance(other, LocDelta):
      return (abs(self.d_ylat - other.d_ylat) < 1e-6) and (abs(self.d_xlon - other.d_xlon) < 1e-6)
    else:
      return False

  def __ne__(self, other):
    return not self.__eq__(other)

  def npArray(self):
    return np.array([self.d_xlon, self.d_ylat])

  def dotProduct(self, other):
    if isinstance(other, LocDelta):
      return (self.d_xlon*other.d_xlon) + (self.d_ylat * other.d_ylat)
    return 

  def getMagnitude(self):
    return math.sqrt((self.d_xlon**2)+(self.d_ylat**2))

  def getUnit(self):
    return self/self.getMagnitude()

  def asPolar(self, degrees=False):
    r = self.getMagnitude()
    theta = math.atan2(self.d_ylat, self.d_xlon)
    if degrees:
      theta = theta * 180 / math.pi
    return r, theta

  def perpindicular(self):
    return LocDelta(d_xlon = self.d_ylat, d_ylat = -self.d_xlon)

  
    
class Observation(object):
  """docstring for Observation"""
  def __init__(self, loc, data, time=None):
    self.loc = loc
    self.data = data
    self.time = time

  def __str__(self):
    if self.time is None:
      return str(self.loc) + " Data: %.5f\n" % (self.data)
    else:
      return str(self.loc) + " Data: %.5f\nTime: %.5f\n" % (self.data, self.time)

  def __iter__(self):
    for item in [self.loc.lon, self.loc.lat, self.time, self.data]:
      yield item
