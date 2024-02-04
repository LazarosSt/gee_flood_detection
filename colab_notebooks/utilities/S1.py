import ee
import math


def power_to_db(raster):
  """
  Description:
    Converts pixel values from power scale to dB scale.

  Arguments:
    raster (ee.Image): The raster with pixel values in power scale.

  Returns:
    The raster with pixel values in dB scale.
  """
  return ee.Image(10).multiply(raster.log10())


def db_to_power(raster):
  """
  Description:
    Converts pixel values from dB scale to power scale.

  Args:
    raster (ee.Image): The raster with pixel values in dB scale.

  Returns:
    The raster with pixel values in power scale.
  """
  return ee.Image(10).pow(raster.divide(10))


def refined_lee(raster):
  """
  Description:
    Applies the refined Lee speckle filter to an raster.

  Arguments:
    raster (ee.Image): The raster to apply the filter on.

  Returns:
    The filtered raster layer.
  """
  def computations(b):
    img = raster.select([b])

    # img must be in natural units, i.e. not in dB!
    # Set up 3x3 kernels
    weights3 = ee.List.repeat(ee.List.repeat(1,3),3)
    kernel3 = ee.Kernel.fixed(3,3, weights3, 1, 1, False)

    mean3 = img.reduceNeighborhood(ee.Reducer.mean(), kernel3)
    variance3 = img.reduceNeighborhood(ee.Reducer.variance(), kernel3)

    # Use a sample of the 3x3 windows inside a 7x7 windows to determine gradients and directions
    sample_weights = ee.List([[0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0], [0,1,0,1,0,1,0], [0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0]])

    sample_kernel = ee.Kernel.fixed(7,7, sample_weights, 3,3, False)

    # Calculate mean and variance for the sampled windows and store as 9 bands
    sample_mean = mean3.neighborhoodToBands(sample_kernel)
    sample_var = variance3.neighborhoodToBands(sample_kernel)

    # Determine the 4 gradients for the sampled windows
    gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs()
    gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs())
    gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs())
    gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs())

    # And find the maximum gradient amongst gradient bands
    max_gradient = gradients.reduce(ee.Reducer.max())

    # Create a mask for band pixels that are the maximum gradient
    gradmask = gradients.eq(max_gradient)

    # duplicate gradmask bands: each gradient represents 2 directions
    gradmask = gradmask.addBands(gradmask)

    # Determine the 8 directions
    directions = sample_mean.select(1).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(7))).multiply(1)
    directions = directions.addBands(sample_mean.select(6).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2))
    directions = directions.addBands(sample_mean.select(3).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3))
    directions = directions.addBands(sample_mean.select(0).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4))
    # The next 4 are the not() of the previous 4
    directions = directions.addBands(directions.select(0).Not().multiply(5))
    directions = directions.addBands(directions.select(1).Not().multiply(6))
    directions = directions.addBands(directions.select(2).Not().multiply(7))
    directions = directions.addBands(directions.select(3).Not().multiply(8))

    # Mask all values that are not 1-8
    directions = directions.updateMask(gradmask)

    # "collapse" the stack into a singe band image (due to masking, each pixel has just one value (1-8) in it's directional band, and is otherwise masked)
    directions = directions.reduce(ee.Reducer.sum())

    sample_stats = sample_var.divide(sample_mean.multiply(sample_mean))

    # Calculate localNoiseVariance
    sigmaV = sample_stats.toArray().arraySort().arraySlice(0,0,5).arrayReduce(ee.Reducer.mean(), [0])

    # Set up the 7*7 kernels for directional statistics
    rect_weights = ee.List.repeat(ee.List.repeat(0,7),3).cat(ee.List.repeat(ee.List.repeat(1,7),4))

    diag_weights = ee.List([[1,0,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,1,0,0,0,0], [1,1,1,1,0,0,0], [1,1,1,1,1,0,0], [1,1,1,1,1,1,0], [1,1,1,1,1,1,1]])

    rect_kernel = ee.Kernel.fixed(7,7, rect_weights, 3, 3, False)
    diag_kernel = ee.Kernel.fixed(7,7, diag_weights, 3, 3, False)

    # Create stacks for mean and variance using the original kernels. Mask with relevant direction.
    dir_mean = img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1))
    dir_var = img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1))

    dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)))
    dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)))

    # and add the bands for rotated kernels
    for i in range(1, 4):
      dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
      dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
      dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))
      dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))

    # "collapse" the stack into a single band image (due to masking, each pixel has just one value in it's directional band, and is otherwise masked)
    dir_mean = dir_mean.reduce(ee.Reducer.sum())
    dir_var = dir_var.reduce(ee.Reducer.sum())

    # A finally generate the filtered value
    varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV)).divide(sigmaV.add(1.0))

    b = varX.divide(dir_var)

    return dir_mean.add(b.multiply(img.subtract(dir_mean))) \
      .arrayProject([0])                                    \
      .arrayFlatten([["sum"]])                              \
      .float()

  bandNames = raster.bandNames()
  raster = db_to_power(raster)

  result = ee.ImageCollection(bandNames.map(computations)).toBands().rename(bandNames)
  return power_to_db(ee.Image(result))


def slope_correction(raster, dem):
  """
  Description:
    Performs slope correction on the input raster using a digital elevation model (DEM).

  Arguments:
    raster (ee.Image): The raster to perform slope correction on.
    dem (ee.Image): The digital elevation model used for correction.

  Returns:
    The slope-corrected raster layer.
  """
  # Obtain the geometry of the image.
  imgGeom = raster.geometry()
  # Clip the extent of the dem layer.
  srtm = ee.Image(dem).clip(imgGeom)
  # Convert pixel values from dB scale to linear scale.
  sigma0Pow = db_to_power(raster)

  # 2.1.1 Radar geometry
  # Compute the mean aspect of the terrain over the region covered by the raster.
  theta_i = raster.select("angle")
  phi_i = ee.Terrain.aspect(theta_i)                                        \
    .reduceRegion(ee.Reducer.mean(), theta_i.get("system:footprint"), 1000) \
    .get("aspect")

  # 2.1.2 Terrain geometry
  # Calculate the slope and aspect of the terrain over the region covered by the raster.
  alpha_s = ee.Terrain.slope(srtm).select("slope")
  phi_s = ee.Terrain.aspect(srtm).select("aspect")

  # 2.1.3 Model geometry
  # reduce to 3 angle
  phi_r = ee.Image.constant(phi_i).subtract(phi_s)

  # Perform some mathematical conversions
  # (convert pixel values from degrees to radians).
  phi_rRad = phi_r.multiply(math.pi / 180)
  alpha_sRad = alpha_s.multiply(math.pi / 180)
  theta_iRad = theta_i.multiply(math.pi / 180)
  ninetyRad = ee.Image.constant(90).multiply(math.pi / 180)

  # slope steepness in range (eq. 2)
  alpha_r = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()

  # slope steepness in azimuth (eq 3)
  alpha_az = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()

  # local incidence angle (eq. 4)
  theta_lia = (alpha_az.cos().multiply((theta_iRad.subtract(alpha_r)).cos())).acos()
  theta_liaDeg = theta_lia.multiply(180 / math.pi)
  # 2.2
  # Gamma_nought_flat
  gamma0 = sigma0Pow.divide(theta_iRad.cos())
  gamma0dB = ee.Image.constant(10).multiply(gamma0.log10())
  ratio_1 = gamma0dB.select("VV").subtract(gamma0dB.select("VH"))

  # Volumetric Model
  nominator = (ninetyRad.subtract(theta_iRad).add(alpha_r)).tan()
  denominator = (ninetyRad.subtract(theta_iRad)).tan()
  volModel = (nominator.divide(denominator)).abs()

  # apply model
  gamma0_Volume = gamma0.divide(volModel)
  gamma0_VolumeDB = ee.Image.constant(10).multiply(gamma0_Volume.log10())

  # we add a layover/shadow maskto the original implmentation
  # layover, where slope > radar viewing angle
  alpha_rDeg = alpha_r.multiply(180 / math.pi)
  layover = alpha_rDeg.lt(theta_i)

  # shadow where LIA > 90
  shadow = theta_liaDeg.lt(85)

  # calculate the ratio for RGB vis
  ratio = gamma0_VolumeDB.select("VV").subtract(gamma0_VolumeDB.select("VH"))

  output = gamma0_VolumeDB.addBands(ratio).addBands(alpha_r).addBands(phi_s).addBands(theta_iRad) \
    .addBands(layover).addBands(shadow).addBands(gamma0dB).addBands(ratio_1)

  return raster.addBands(
    output.select(["VV", "VH"], ["VV", "VH"]),
    None,
    True
  )