# Renaming camera trap imagery with Excel

Each camera trap project uses a different naming convention. A standardised naming convention has therefore been developed and applied to all images. 

Examples of the naming format are provided below:
### BE-LE_40_16-05-23_160000.JPG	
### SE-KL_111_02_03_24_120000.JPG

The first section is the project code, beginning with the two-letter ISO 3166-1 alpha-2 country code, followed by a hyphen and a two-letter code representing the project region, creating a unique four-letter project identifier.
The next component is the deployment ID assigned to each individual camera trap. 
A standard incremental numbering system has been applied, which, in combination with the project code, creates a unique identifier. 

The numbering system is arbitrary and may be derived either from the original deployment IDs in alphabetical order or from project start order.
For the date section, write the date as DD&MM&YY rather than dd-mm-yy or dd/mm/yy, as these formats can trigger regional settings in Excel and cause automatic reformatting. 
Using & (or another symbol that will not appear within the name) allows for a simple search-and-replace operation after concatenation.
For the time section, add the six-digit format %H%M%S to indicate image capture time. 


If the camera trap did not capture the image exactly on the hour, round to the nearest hour to avoid unnecessary complexity. The time value is used to distinguish between different timelapse images from the same deployment. 
If required, the true acquisition time can be included as a separate column.
The renaming process was performed in Excel using three columns of information: deployment_id (including the project code), date, and time. 
The CONCATENATE function was then used to join these values with an underscore separator and append the file extension .JPG.
It is important to retain an index that maps the original filename to the new filename to ensure traceability.
