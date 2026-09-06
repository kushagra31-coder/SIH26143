import os
import json
import argparse
from datetime import datetime, timedelta, timezone
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_hindcast(seed_path, output_path):
    if not os.path.exists(seed_path):
        logging.error(f"Seed file not found: {seed_path}")
        return False
        
    with open(seed_path, 'r') as f:
        seed = json.load(f)
        
    lon = seed["lon"]
    lat = seed["lat"]
    time_str = seed["time"]
    start_time = datetime.fromisoformat(time_str.replace("Z", ""))
    
    # Grounding occurred July 25, 2020 19:25 LT (approx 15:25 UTC)
    end_time = datetime(2020, 7, 25, 15, 25)
    
    try:
        # Import OpenDrift only when needed to save time if running checks
        from opendrift.models.openoil import OpenOil
        o = OpenOil(loglevel=20)
        
        # Add readers
        logging.info("Adding readers...")
        # For MVP, try to use global readers if available, else rely on built-in fallback/constant
        o.add_readers_from_list(['global_landmask'])
        
        cmems_user = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
        cmems_pass = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")
        
        used_synthetic_forcing = False
        
        if cmems_user and cmems_pass:
            logging.info("CMEMS credentials found! Setting up real wind/current forcing...")
            try:
                import subprocess
                import sys
                from opendrift.readers import reader_netCDF_CF_generic
                
                # Download currents (modern dataset ID: cmems_mod_glo_phy_my_0.083deg_P1D-m)
                logging.info("Downloading CMEMS currents...")
                subprocess.run([
                    "copernicusmarine", "subset", 
                    "-i", "cmems_mod_glo_phy_my_0.083deg_P1D-m", 
                    "-x", "50.0", "-X", "65.0", 
                    "-y", "-26.0", "-Y", "-15.0", 
                    "-t", "2020-07-24", "-T", "2020-08-07", 
                    "-v", "uo", "-v", "vo", 
                    "-f", "cmems_currents.nc"
                ], check=True)
                
                # Download winds (modern dataset ID: cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H)
                logging.info("Downloading CMEMS winds...")
                subprocess.run([
                    "copernicusmarine", "subset", 
                    "-i", "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H", 
                    "-x", "50.0", "-X", "65.0", 
                    "-y", "-26.0", "-Y", "-15.0", 
                    "-t", "2020-07-24", "-T", "2020-08-07", 
                    "-v", "eastward_wind", "-v", "northward_wind", 
                    "-f", "cmems_winds.nc"
                ], check=True)
                
                r_currents = reader_netCDF_CF_generic.Reader("cmems_currents.nc")
                r_winds = reader_netCDF_CF_generic.Reader("cmems_winds.nc")
                o.add_reader([r_currents, r_winds])
                logging.info("Real CMEMS readers added successfully.")
            except Exception as e:
                logging.critical(f"Failed to load real CMEMS data! Make sure you accepted the Copernicus Marine Terms of Service for these datasets: {e}")
                logging.warning("Falling back to synthetic constant wind/current.")
                used_synthetic_forcing = True
        else:
            logging.warning("No CMEMS credentials found in environment. Using synthetic constant forcing.")
            used_synthetic_forcing = True
            
        if used_synthetic_forcing:
            from opendrift.readers import reader_constant
            # Constant wind towards NW (from SE)
            r = reader_constant.Reader(x_wind=-5, y_wind=5, x_sea_water_velocity=-0.2, y_sea_water_velocity=0.2)
            o.add_reader(r)
        
        logging.info(f"Seeding oil at {lon}, {lat} at {start_time}")
        o.seed_elements(lon=lon, lat=lat, time=start_time, number=1000, radius=seed.get("uncertainty_radius", 1000))
        
        logging.info(f"Running backtracking hindcast to {end_time}")
        # Run backwards
        o.run(end_time=end_time, time_step=-timedelta(hours=1), time_step_output=timedelta(hours=6))
        
        # Extract the final positions (which is the origin since we are backtracking)
        try:
            # Modern versions (using xarray dataset)
            lons = o.result.lon.values
            lats = o.result.lat.values
        except AttributeError:
            try:
                # Older dictionary-based history
                lons = o.history['lon']
                lats = o.history['lat']
            except (AttributeError, TypeError, KeyError):
                # Fallback to property getter
                lons = o.get_property('lon')[0]
                lats = o.get_property('lat')[0]
        
        # The last step in the array is the state at end_time (the origin)
        final_lons = lons[:, -1]
        final_lats = lats[:, -1]
        
        # Calculate centroid of the backtracked particles
        origin_lon = final_lons.mean()
        origin_lat = final_lats.mean()
        
        logging.info(f"Backtracked Origin Centroid: Lon {origin_lon:.4f}, Lat {origin_lat:.4f}")
        
        # Save output
        output_data = {
            "origin_lon": float(origin_lon),
            "origin_lat": float(origin_lat),
            "origin_time": end_time.isoformat() + "Z",
            "particles": {
                "lons": final_lons.tolist(),
                "lats": final_lats.tolist()
            }
        }
        
        if used_synthetic_forcing:
            output_data["SYNTHETIC_TEST_DATA"] = True
            logging.warning("This OpenDrift run was fed synthetic wind/current data and is NOT a physical result.")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        logging.info(f"Saved drift origin output to {output_path}")
        return True
        
    except ImportError:
        logging.critical("=========================================================================")
        logging.critical("                           [SYNTHETIC_TEST_DATA]                         ")
        logging.critical(" OpenDrift/Cartopy is not installed (blocked by native Windows C++ deps).")
        logging.critical(" INJECTING SYNTHETIC DRIFT ORIGIN for pipeline/UI validation purposes.   ")
        logging.critical(" THIS IS NOT A PHYSICAL RESULT.                                          ")
        logging.critical("=========================================================================")
        
        output_data = {
            "origin_lon": 57.745,
            "origin_lat": -20.438,
            "origin_time": end_time.isoformat() + "Z",
            "particles": {
                "lons": [57.745],
                "lats": [-20.438]
            },
            "SYNTHETIC_TEST_DATA": True
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        logging.info(f"[SYNTHETIC_TEST_DATA] Saved synthetic drift origin output to {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error running OpenDrift: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenOil backward drift hindcast")
    parser.add_argument("--seed", required=True, help="Path to opendrift_seed.json")
    parser.add_argument("--output", required=True, help="Path to output origin.json")
    
    args = parser.parse_args()
    run_hindcast(args.seed, args.output)
