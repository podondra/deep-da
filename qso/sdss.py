from os import path


def get_spec_filename(spec):
    plate = spec["plate"]
    mjd = spec["mjd"]
    fiberid = spec["fiberid"]
    filename_str = "spec-{:04d}-{:05d}-{:04d}.fits"
    return filename_str.format(plate, mjd, fiberid)


def get_spec_filepath(spec, filename):
    plate = spec["plate"]
    return path.join("{:04d}".format(plate), filename)
