# duplicate the slurm_snowlines_SITE-ID.bash file and save with new site ID

old_id = "SITE-ID"
new_ids = ["01.00032", "01.00033", "01.00037", "01.00038", "01.00046", 
           "01.00312", "01.00566", "01.00570", "01.00576", "01.00675"]

# Load input file
input_fn = "/Users/raineyaberle/Research/PhD/snow_cover_mapping/snow-cover-mapping/scripts/slurm_RGI60-SITE-ID.bash"
with open(input_fn, "r") as input_file:
    content = input_file.read()

# Iterate over new_ids
for new_id in new_ids:

    # Perform the replacement
    new_content = content.replace(old_id, new_id)

    # Define output file name
    output_fn = input_fn.replace(old_id, new_id)

    # Open the input file for writing
    with open(output_fn, "w") as output_file:
        output_file.write(new_content)
    print("Saved new bash file for " + new_id)
