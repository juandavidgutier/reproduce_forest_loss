# ####################################### LOAD REQUIRED LIBRARIES ############################################# #
import os, csv
import time
import gdal, osr, ogr
import numpy as np
#import baumiTools as bt
from joblib import Parallel, delayed
from tqdm import tqdm
# ####################################### SET TIME-COUNT ###################################################### #
if __name__ == '__main__':
    starttime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("--------------------------------------------------------")
    print("Starting process, time:" +  starttime)
    print("")
# ####################################### FOLDER PATHS AND BASIC VARIABLES FOR PROCESSING ##################### #
# Tiles from the global forest loss dataset are being merged into a virtual raster dataset (.vrt) using the gdal_buildvrt function
    rootFolder = "D:/Forest/"
    pol_shp = rootFolder + "colombia_shp/Municipios wgs84_Disolv.shp"
    out_csv = rootFolder + "DataSummaries/Summaries.csv"
    forest = rootFolder + "Forest2000.vrt"
    gain = rootFolder + "ForestGain.vrt"
    loss = rootFolder + "ForestLoss.vrt"
    nPackages = 10
    nr_cores = 12
# ####################################### PROCESSING ########################################################## #
# (1) Build job list
    jobList = []
    # Get the number of total features in the shapefile
    eco = ogr.Open(pol_shp)
    ecoLYR = eco.GetLayer()
    nFeat = ecoLYR.GetFeatureCount()
    # Create a list of UIDs and subdivide the into smaller chunks
    featIDs = list(range(1,nFeat+1, 1))
    packageSize = int(nFeat / nPackages)
    #
    IDlist = [featIDs[i * packageSize:(i + 1) * packageSize] for i in range((len(featIDs) + packageSize - 1) // packageSize )]
    # Now build the jobs and append to job list
    for chunk in IDlist:
        job = {'ids': chunk,
               'shp_path': pol_shp,
               'forest_raster': forest,
               'gain_raster': gain,
               'loss_raster': loss}
        jobList.append(job)
        
        
        
# (2) Build Worker_Function


#################################################
#Function copied from the repository baumiTools, because the module can be installed in Python
from osgeo import ogr

def CopyToMem(path):
    drvMemV = ogr.GetDriverByName('Memory')
    f_open = drvMemV.CopyDataSource(ogr.Open(path),'')
    return f_open
#################################################


def SumFunc(job):
    # Define the drivers that we need for creating the summaries
        drvMemV = ogr.GetDriverByName('Memory')
        drvMemR = gdal.GetDriverByName('MEM')         
        # Load the shapefile into mem, get the layer and subset by the IDs that are in the chunk
        shpMem = CopyToMem(job['shp_path'])    
        lyr = shpMem.GetLayer()
        idSubs = job['ids']
        lyr.SetAttributeFilter("Codigo IN {}".format(tuple(idSubs)))
        # Create coordinate transformation rule
        pol_SR = lyr.GetSpatialRef()
        # Define the output-list that we want to return
        outList = []
        # Now loop through the selected features in our lyr
        feat = lyr.GetNextFeature()
        while feat:
    # Get needed properties from the SHP-File, the take geometry, and transform to Target-EPSG
            # UID Info
            UID = feat.GetField("Codigo")
    # Instantiate output and take the geometry of the feature, transform it to our epsg
            vals = [UID]
            geom = feat.GetGeometryRef()
    # Rasterize the geometry, pixelSize is 30m
        # Create new SHP-file in memory to which we copy the geometry
            geom_shp = drvMemV.CreateDataSource('')
            geom_lyr = geom_shp.CreateLayer('geom', pol_SR, geom_type=ogr.wkbMultiPolygon)
            geom_lyrDefn = geom_lyr.GetLayerDefn()
            geom_feat = ogr.Feature(geom_lyrDefn)
            geom_feat.SetGeometry(geom)
            geom_lyr.CreateFeature(geom_feat)
        # Check if the geometry we are processing is larger than 1x1 pixel
            x_min, x_max, y_min, y_max = geom_lyr.GetExtent()
            x_res = int((x_max - x_min) / 30)
            y_res = int((y_max - y_min) / 30)
        
            geom_ras = drvMemR.Create('', x_res, y_res, gdal.GDT_Byte)
            geom_ras.SetProjection(pol_SR.ExportToWkt())
            geom_ras.SetGeoTransform((x_min, 30, 0, y_max, 0, -30))

            gdal.RasterizeLayer(geom_ras, [1], geom_lyr, burn_values=[1])
        # Reproject the Hansen-Rasters "into" the geometry-raster
            def ReprojectRaster(valRaster, GEOMraster):
                vasRaster_sub = drvMemR.Create('', GEOMraster.RasterXSize, GEOMraster.RasterYSize, 1, gdal.GDT_Byte)
                vasRaster_sub.SetGeoTransform(GEOMraster.GetGeoTransform())
                vasRaster_sub.SetProjection(GEOMraster.GetProjection())
                gdal.ReprojectImage(valRaster, vasRaster_sub, valRaster.GetProjection(), GEOMraster.GetProjection(), gdal.GRA_NearestNeighbour)
                return vasRaster_sub
            forest = ReprojectRaster(gdal.Open(job['forest_raster']), geom_ras)
            loss = ReprojectRaster(gdal.Open(job['loss_raster']), geom_ras)
            gain = ReprojectRaster(gdal.Open(job['gain_raster']), geom_ras)
        # Open all rasters into np-Arrays
            geom_np = geom_ras.GetRasterBand(1).ReadAsArray(0, 0, x_res, y_res)
            forest_np = forest.GetRasterBand(1).ReadAsArray(0, 0, x_res, y_res)
            loss_np = loss.GetRasterBand(1).ReadAsArray(0, 0, x_res, y_res)
            gain_np = gain.GetRasterBand(1).ReadAsArray(0, 0, x_res, y_res)
        # Now extract the summaries
            # Forest 2000 --> 25% canopy
            forest_np_25 = np.where((geom_np == 1) & (forest_np >= 25), 1, 0)
            forest_np_25_mask = np.where((geom_np == 1) & (forest_np >= 25), 1, 0)
            forest_np_25 = forest_np_25.astype(np.uint8)
            f25 = forest_np_25.sum() * 30 * 30 / 10000
            vals.append(format(f25, '.5f'))
            # Loss per year
            for yr in range(1, 18, 1):
                loss_np_yr = np.where((geom_np == 1) & (loss_np == yr) & (forest_np_25_mask == 1), 1, 0)
                loss_np_yr = loss_np_yr.astype(np.uint8)
                loss_yr = loss_np_yr.sum() * 30 * 30 / 10000
                vals.append(format(loss_yr, '.5f'))
            # gain
            gain_np_mask = np.where((geom_np == 1) & (gain_np == 1) & (forest_np_25_mask == 1), 1, 0)
            gain_np_mask = gain_np_mask.astype(np.uint8)
            gn = gain_np_mask.sum() * 30 * 30 / 10000
            vals.append(format(gn, '.5f'))
        # Append the values to the output-DS, then take the next feature
            outList.append(vals)
            feat = lyr.GetNextFeature()
    # return the outList as output from the function
        return outList
    


    
# (3) Execute the Worker_Funtion parallel
    job_results = Parallel(n_jobs=nr_cores)(delayed(SumFunc)(i) for i in tqdm(jobList))
    

# (4) Merge the different packages back together into one dataset, instantiate colnames first
    print("Merge Outputs")
    outDS = [["Codigo",
              "F2000_Ha_th25", "FL2001_Ha", "FL2002_Ha", "FL2003_Ha", "FL2004_Ha", "FL2005_Ha",
              "FL2006_Ha", "FL2007_Ha", "FL2008_Ha", "FL2009_Ha", "FL2010_Ha", "FL2011_Ha", "FL2012_Ha","FL2013_Ha",
              "FL2014_Ha", "FL2015_Ha", "FL2016_Ha", "FL2017_Ha", "FL_2018_Ha", "Gain_Ha"]]
    # Now extract the information from all the evaluations
    # 1st loop --> the different chunks
    for result in job_results:
        # 2nd loop --> all outputs in each chunk
        for out in result:
            outDS.append(out)


# (5) Write all outputs to disc
    print("Write output")
    with open(out_csv, "w") as theFile:
        csv.register_dialect("custom", delimiter = ",", skipinitialspace = True, lineterminator = '\n')
        writer = csv.writer(theFile, dialect = "custom")
        for element in outDS:
            writer.writerow(element)
# ####################################### END TIME-COUNT AND PRINT TIME STATS################################## #
    print("")
    endtime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("--------------------------------------------------------")
    print("--------------------------------------------------------")
    print("start: " + starttime)
    print("end: " + endtime)
    print("")