import ee


def mask_clouds(raster, cloudProbability):
  """
  Description:
    Masks clouds in a raster based on the specified cloud probability.

  Arguments:
    raster (ee.Image): The raster containing the cloud mask.
    cloudProbability (int): The threshold probability for cloud masking.

  Returns:
    The cloud-masked raster layer.
  """
  clouds = ee.Image(raster.get("cloud_mask")).select("probability")
  isNotCloud = clouds.lt(cloudProbability)
  return raster.updateMask(isNotCloud)


def mask_edges(raster):
  """
  Description:
    Masks edges of a raster using masks from different bands.

  Note:
    The masks for the 10m bands sometimes do not exclude bad data at scene edges,
    so we apply masks from the 20m and 60m bands as well.

  Arguments:
    raster (ee.Image): The raster to mask.

  Returns:
    The masked raster layer.

  Example:
    raster = ee.Image("COPERNICUS/S2_CLOUD_PROBABILITY/20190301T000239_20190301T000238_T55GDP")
  """
  edgesMask = raster.select("B8A").mask().updateMask(raster.select("B9").mask())
  return raster.updateMask(edgesMask)