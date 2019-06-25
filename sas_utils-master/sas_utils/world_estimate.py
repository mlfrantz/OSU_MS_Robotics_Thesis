from .gp_world_model import GPStaticWorldModel, GPTimeVaryingWorldModel, GPComboTimeVaryingWorldModel
from scipy.interpolate import griddata, RegularGridInterpolator
import numpy as np
import pdb, datetime, itertools, random, time, os, math
from .location import Observation, Location, LocDelta
from .utils import initializeNovelty, getBox
from .world import World
from collections import OrderedDict
import deepdish as dd
import matplotlib.pyplot as plt
import matplotlib.ticker as tick


class WorldEstimate(object):
  """docstring for WorldEstimate"""
  def __init__(self, sci_type, snapshot_time, scalar_field, variance_estimate, current_u_field, current_v_field, x_ticks, y_ticks, env_resolution):
    self.science_variable_type = sci_type
    self.snapshot_time = snapshot_time

    self.scalar_field = scalar_field
    self.current_u_field = current_u_field
    self.current_v_field = current_v_field
    self.variance_scalar_field = variance_estimate

    self.x_ticks = x_ticks # km
    self.y_ticks = y_ticks # km

    self.env_resolution = env_resolution # km

  def withinBounds(self, query_loc):
    if query_loc.x < np.min(self.x_ticks):
      return False
    if query_loc.x > np.max(self.x_ticks):
      return False
    if query_loc.y < np.min(self.y_ticks):
      return False
    if query_loc.y > np.max(self.y_ticks):
      return False
    return True

  def getRandomLocationXY(self):
    return Location(xlon=random.choice(self.x_ticks), ylat=random.choice(self.y_ticks))

  def getUVcurrent(self, loc, t=None):
    u_interp = RegularGridInterpolator((self.x_ticks, self.y_ticks), self.current_u_field, fill_value=0.0, bounds_error=False)
    v_interp = RegularGridInterpolator((self.x_ticks, self.y_ticks), self.current_v_field, fill_value=0.0, bounds_error=False)

    current_u_current = u_interp((loc.x, loc.y))
    current_v_current = v_interp((loc.x, loc.y))

    return LocDelta(d_xlon = float(current_u_current), d_ylat = float(current_v_current))

  def store(self, filename, ct=None):
    print('Processing ...')
    time = "t%d" % self.snapshot_time
    depth = 'surface'

    snapshot_dict = OrderedDict()

    snapshot_dict[self.science_variable_type]   = self.scalar_field.flatten()
    snapshot_dict['variance_est']               = self.variance_scalar_field.flatten()
    snapshot_dict['currentslongitudinal']       = self.current_u_field.flatten() # np.ones(self.current_u_field.flatten().shape)
    snapshot_dict['currentslatitudinal']        = self.current_v_field.flatten() # np.zeros(self.current_v_field.flatten().shape)
    snapshot_dict['currentsspeed']              = np.sqrt(np.power(snapshot_dict['currentslongitudinal'],2)+np.power(snapshot_dict['currentslatitudinal'],2))
    snapshot_dict['currentsheading']            = 90.-np.rad2deg(np.arctan2(snapshot_dict['currentslatitudinal'],snapshot_dict['currentslongitudinal']))

    yy, xx = np.meshgrid(self.y_ticks, self.x_ticks)
    snapshot_dict['time'] = time
    snapshot_dict['x_km']                   = xx.flatten()
    snapshot_dict['y_km']                   = yy.flatten()
    snapshot_dict['depth']                  = np.zeros(yy.shape).flatten()

    depth_dict = OrderedDict()
    depth_dict[depth] = snapshot_dict

    res = {}
    res[time] = depth_dict

    print('Storing Local Belief to file %s ...' % filename)
    dd.io.save(filename, res, compression='zlib')

    if ct is not None:
      latlon_locs = [ct.xy2latlon(Location(xlon=x, ylat=y)) for x, y in zip(xx.flatten(), yy.flatten())]
      snapshot_dict['latitude'] = np.array([loc.lat for loc in latlon_locs])
      snapshot_dict['longitude'] = np.array([loc.lon for loc in latlon_locs])
      snapshot_dict['currentsheading'] = 90 - np.rad2deg(np.arctan2(snapshot_dict['currentslatitudinal'],snapshot_dict['currentslongitudinal'])) + math.degrees(ct.rotate_theta)

      snapshot_dict.pop("x_km")
      snapshot_dict.pop("y_km")

      global_filename = filename.replace(os.path.basename(filename), "global_" + os.path.basename(filename))

      depth_dict = OrderedDict()
      depth_dict[depth] = snapshot_dict

      res = {}
      res['belief'] = depth_dict
      print('Storing Global Belief to file %s ...' % global_filename)
      dd.io.save(global_filename, res, compression='zlib')


  @classmethod
  def GPFromObservations(cls, sci_type, sci_obs, u_obs, v_obs, mission_params, global_estimator_params, estimator_params, snapshot_time=time.time()):
    time_scale_factor = 3600.*24.
    env_resolution = global_estimator_params['resolution']
    gp_length_scales = [global_estimator_params['spatial_lengthscale'], global_estimator_params['spatial_lengthscale'], global_estimator_params['temporal_lengthscale']/time_scale_factor]
    gp_init_variance = estimator_params['gp_variance']

    x_ticks = np.arange(0.0, mission_params['width']+env_resolution, env_resolution)
    y_ticks = np.arange(0.0, mission_params['height']+env_resolution, env_resolution)

    x_ticks = x_ticks - np.max(x_ticks)/2.
    y_ticks = y_ticks - np.max(y_ticks)/2.

    if len(sci_obs) > 0:
      sci_gp = GPTimeVaryingWorldModel(sci_obs, gp_init_variance, gp_length_scales, x_ticks, y_ticks, time_scale_factor)
      sci_estimate, variance_estimate = sci_gp.getGPModel(snapshot_time, x_ticks, y_ticks)
    else:
      sci_estimate = np.zeros((len(x_ticks), len(y_ticks)))
      variance_estimate = np.zeros((len(x_ticks), len(y_ticks)))

    if len(u_obs) > 0:
      u_gp = GPTimeVaryingWorldModel(u_obs, gp_init_variance, gp_length_scales, x_ticks, y_ticks, time_scale_factor)
      u_estimate, _ = u_gp.getGPModel(snapshot_time, x_ticks, y_ticks)
    else:
      u_estimate = np.zeros((len(x_ticks), len(y_ticks)))

    if len(v_obs) > 0:
      v_gp = GPTimeVaryingWorldModel(v_obs, gp_init_variance, gp_length_scales, x_ticks, y_ticks, time_scale_factor)
      v_estimate, _ = v_gp.getGPModel(snapshot_time, x_ticks, y_ticks)
    else:
      v_estimate = np.zeros((len(x_ticks), len(y_ticks)))

    return cls(sci_type, snapshot_time, sci_estimate, variance_estimate, u_estimate, v_estimate, x_ticks, y_ticks, env_resolution)

  @classmethod
  def NearestNeighborFromObservations(cls, sci_type, sci_obs, u_obs, v_obs, mission_params, global_estimator_params, estimator_params, snapshot_time=time.time()):
    x_ticks = np.arange(0.0, mission_params['width']+env_resolution, env_resolution)
    y_ticks = np.arange(0.0, mission_params['height']+env_resolution, env_resolution)

    x_ticks = x_ticks - np.max(x_ticks)/2.
    y_ticks = y_ticks - np.max(y_ticks)/2.

    gp_prior = global_estimator_params['prior']
    variance = np.mean(global_estimator_params['length_scales'][:2])

    yy,xx = np.meshgrid(y_ticks,x_ticks)

    if len(sci_obs) > 0:
      sci_pts = np.array([[o.loc.lon, o.loc.lat] for o in sci_obs])
      sci_data = np.array([o.data for o in sci_obs])
      sci_estimate = griddata(sci_pts, sci_data, (xx,yy), fill_value=gp_prior, method='nearest')
      variance_estimate = initializeNovelty(sci_obs, xx, yy, variance, global_estimator_params['temporal_lengthscale'], global_estimator_params['obs_novelty_magnitude'])
    else:
      sci_estimate = np.zeros((len(x_ticks), len(y_ticks)))
      variance_estimate = np.zeros((len(x_ticks), len(y_ticks)))

    if len(u_obs) > 0:
      u_pts = np.array([[o.loc.lon, o.loc.lat] for o in u_obs])
      u_data = np.array([o.data for o in u_obs])
      u_estimate = griddata(u_pts, u_data, (xx,yy), fill_value=gp_prior, method='nearest')
    else:
      u_estimate = np.zeros((len(x_ticks), len(y_ticks)))

    if len(v_obs) > 0:
      v_pts = np.array([[o.loc.lon, o.loc.lat] for o in v_obs])
      v_data = np.array([o.data for o in v_obs])
      v_estimate = griddata(v_pts, v_data, (xx,yy), fill_value=gp_prior, method='nearest')
    else:
      v_estimate = np.zeros((len(x_ticks), len(y_ticks)))

    return cls(sci_type, snapshot_time, sci_estimate, variance_estimate, u_estimate, v_estimate, x_ticks, y_ticks, env_resolution)

  @classmethod
  def comboFromObservations(cls, sci_type, sci_obs, u_obs, v_obs, mission_params, global_estimator_params, estimator_params, snapshot_time=time.time()):
    time_scale_factor = 3600.*24.
    env_resolution = global_estimator_params['resolution']
    gp_length_scales = [global_estimator_params['spatial_lengthscale'], global_estimator_params['spatial_lengthscale'], global_estimator_params['temporal_lengthscale']/time_scale_factor]
    gp_init_variance = estimator_params['gp_variance']

    x_ticks = np.arange(0.0, mission_params['width']+env_resolution, env_resolution)
    y_ticks = np.arange(0.0, mission_params['height']+env_resolution, env_resolution)

    x_ticks = x_ticks - np.max(x_ticks)/2.
    y_ticks = y_ticks - np.max(y_ticks)/2.

    if len(sci_obs) > 0:
      sci_gp = GPComboTimeVaryingWorldModel(sci_obs, gp_init_variance, gp_length_scales, x_ticks, y_ticks, time_scale_factor)
      sci_estimate, variance_estimate = sci_gp.getGPModel(snapshot_time, x_ticks, y_ticks)
    else:
      sci_estimate = np.zeros((len(x_ticks), len(y_ticks)))
      variance_estimate = np.zeros((len(x_ticks), len(y_ticks)))

    if len(u_obs) > 0:
      u_gp = GPComboTimeVaryingWorldModel(u_obs, gp_init_variance, gp_length_scales, x_ticks, y_ticks, time_scale_factor)
      u_estimate, _ = u_gp.getGPModel(snapshot_time, x_ticks, y_ticks)
    else:
      u_estimate = np.zeros((len(x_ticks), len(y_ticks)))

    if len(v_obs) > 0:
      v_gp = GPComboTimeVaryingWorldModel(v_obs, gp_init_variance, gp_length_scales, x_ticks, y_ticks, time_scale_factor)
      v_estimate, _ = v_gp.getGPModel(snapshot_time, x_ticks, y_ticks)
    else:
      v_estimate = np.zeros((len(x_ticks), len(y_ticks)))

    return cls(sci_type, snapshot_time, sci_estimate, variance_estimate, u_estimate, v_estimate, x_ticks, y_ticks, env_resolution)

  @classmethod
  def fromFile(cls, file_path, mission_params):
    we_dict = dd.io.load(file_path)
    snapshot_time_key = we_dict.keys()[0]
    we_dict = we_dict[snapshot_time_key]
    we_dict = we_dict[we_dict.keys()[0]] # Yes this is here twice to get through the time and depth keys

    snapshot_time = int(snapshot_time_key[1:])
    x_ticks = we_dict['x_km']
    y_ticks = we_dict['y_km']

    num_x_ticks = len(np.unique(x_ticks))
    num_y_ticks = len(np.unique(y_ticks))

    assert num_y_ticks * num_x_ticks == len(we_dict[mission_params['science_variable']]), "ERROR: Dimension Mismatch on Reconstructed Scalar Field"

    x_ticks = x_ticks.reshape(num_x_ticks, num_y_ticks)[:,0]
    y_ticks = y_ticks.reshape(num_x_ticks, num_y_ticks)[0]

    scalar_field = we_dict[mission_params['science_variable']].reshape(num_x_ticks, num_y_ticks)
    variance_estimate = we_dict['variance_est'].reshape(num_x_ticks, num_y_ticks)
    current_u_field = we_dict['currentslongitudinal'].reshape(num_x_ticks, num_y_ticks)
    current_v_field = we_dict['currentslatitudinal'].reshape(num_x_ticks, num_y_ticks)

    assert scalar_field.shape == (num_x_ticks, num_y_ticks)
    assert variance_estimate.shape == (num_x_ticks, num_y_ticks)
    assert current_u_field.shape == (num_x_ticks, num_y_ticks)
    assert current_v_field.shape == (num_x_ticks, num_y_ticks)

    cell_x_size = np.mean([x_ticks[idx+1]-x_ticks[idx] for idx, _ in enumerate(x_ticks[:-1])])
    cell_y_size = np.mean([y_ticks[idx+1]-y_ticks[idx] for idx, _ in enumerate(y_ticks[:-1])])

    if cell_x_size != cell_y_size:
      print("Warning: Environmental Mismatch in X and Y dimensions!")

    return cls(mission_params['science_variable'], snapshot_time, scalar_field, variance_estimate, current_u_field, current_v_field, x_ticks, y_ticks,  cell_x_size)


  def draw(self, ax, block=True, show=False, cbar_max=None, cbar_min=None, quiver_stride=7):

    if cbar_min is None:
      cbar_min = np.min(self.scalar_field)
    if cbar_max is None:
      cbar_max = np.max(self.scalar_field)

    num_format  = '%.0f'
    formatter = tick.FormatStrFormatter(num_format)

    CS = plt.pcolor(self.x_ticks, self.y_ticks, self.scalar_field.transpose(), cmap='Greys', vmin=cbar_min, vmax=cbar_max)

    quiver = plt.quiver(self.x_ticks[::quiver_stride], self.y_ticks[::quiver_stride], self.current_u_field.transpose()[::quiver_stride, ::quiver_stride], self.current_v_field.transpose()[::quiver_stride, ::quiver_stride])
    quiver_key = plt.quiverkey(quiver, 0.95, 1.05, 0.2, "0.2 m/s", labelpos='E', coordinates='axes')

    ax.get_xaxis().set_major_formatter(formatter)
    ax.get_yaxis().set_major_formatter(formatter)
    plt.ylim([np.min(self.x_ticks), np.max(self.x_ticks)])
    plt.xlim([np.min(self.y_ticks), np.max(self.y_ticks)])
    cbar = plt.colorbar(CS, format='%.1f')
    plt.title("Estimate Field")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    cbar.set_label(self.science_variable_type)
    ax.axis('scaled')

    if show:
      plt.show(block)


  def drawVariance(self, ax, block=True, show=False, cbar_max=None, cbar_min=None, quiver_stride=7):

    if cbar_min is None:
      cbar_min = np.min(self.variance_scalar_field)
    if cbar_max is None:
      cbar_max = np.max(self.variance_scalar_field)

    num_format  = '%.0f'
    formatter = tick.FormatStrFormatter(num_format)

    CS = plt.pcolor(self.x_ticks, self.y_ticks, self.variance_scalar_field.transpose(), cmap='Greys', vmin=cbar_min, vmax=cbar_max)

    ax.get_xaxis().set_major_formatter(formatter)
    ax.get_yaxis().set_major_formatter(formatter)
    plt.ylim([np.min(self.x_ticks), np.max(self.x_ticks)])
    plt.xlim([np.min(self.y_ticks), np.max(self.y_ticks)])
    cbar = plt.colorbar(CS, format='%.1f')
    plt.title("Estimate Variance")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    cbar.set_label(self.science_variable_type)
    ax.axis('scaled')

    if show:
      plt.show(block)


  def getRandomLocation(self):
    return [random.choice(self.x_ticks), random.choice(self.y_ticks)]
