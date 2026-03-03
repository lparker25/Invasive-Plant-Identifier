# Changes Made: Multi-Species Training & Unknown Plant Detection

## Overview
The training system has been enhanced to support training on multiple species simultaneously, and unknown plant detection has been improved with configurable thresholds and visual feedback.

## Key Changes

### 1. **Training Interface - Multi-Species Support**

#### Previous Behavior
- Single species at a time
- Text input for only one species name
- Limited data organization options

#### New Behavior
- **Two-step process for better organization:**
  
  **Step 1: Upload Images**
  - Upload images for a single species (with species name)
  - OR upload a zip file with folders organized by species
  - Example zip structure: `species1/image1.jpg`, `species2/image2.jpg`
  
  **Step 2: Select Species & Train**
  - View all available species with image counts
  - Multi-select dropdown to choose multiple species for training
  - All selected species train together in a single training run
  - Configurable epochs slider (1-20 epochs, default 5)

#### How It Works
```
data/
  ├── Species1/
  │   ├── image1.jpg
  │   └── image2.jpg
  ├── Species2/
  │   ├── image1.jpg
  │   └── image2.jpg
  └── Species3/
      └── image1.jpg
```

When training, Select Species1 + Species2 + Species3 and the model is trained on all three simultaneously.

### 2. **Unknown Plant Detection**

#### Previous Behavior
- Fixed confidence threshold of 0.95
- No visual feedback for unknown detections

#### New Behavior
- **Configurable confidence threshold slider** (0.0 - 1.0)
  - Default: 0.95 (95% confidence required for species identification)
  - Lower threshold = more "unknown" detections
  - Higher threshold = stricter species identification
  
- **Visual feedback:**
  - Unknown detections show warning icon (⚠️) and highlighted message
  - "UNKNOWN SPECIES - Confidence below threshold" message
  - Clear distinction between identified and unknown plants

#### How It Works
1. Model makes a prediction with confidence score
2. If confidence < threshold: marked as "unknown"
3. If confidence >= threshold: marked as the predicted species
4. All detections logged to database

### 3. **Database Enhancements**

#### New Features
- **Detection Statistics in Visualizations**
  - Shows count of "Known Species Detections"
  - Shows count of "Unknown Species Detections"
  - Easy tracking of identification success rate

- **Species Management**
  - Cannot mark "unknown" as invasive (it's just low-confidence detections)
  - Clear info message explaining "unknown" species

#### Statistics Displayed
- Known species: Successfully identified plants (confidence >= threshold)
- Unknown: Plants below confidence threshold (requires more training data)

## Technical Details

### Model Architecture
- The underlying PyTorch ResNet18 model already supports any number of output classes
- Final layer automatically resizes to match number of species
- Training process uses PyTorch ImageFolder format which naturally supports multiple class folders

### Label Management
- Labels are managed in `labels.json`
- Automatically updated when new species are added
- Maps species names to numeric indices for the model

### Database
- Unknown plants stored with `species = "unknown"`
- `is_invasive` field always 0 for unknown species
- Can be edited in database viewer or filtered in analysis

## Usage Examples

### Example 1: Training on Multiple Species
1. Go to Training tab
2. Upload zip: `species_images.zip` containing:
   ```
   Bluebonnet/image1.jpg
   Bluebonnet/image2.jpg
   YFH/image1.jpg
   YFH/image2.jpg
   Cotton/image1.jpg
   ```
3. Don't enter species name (zip has folders)
4. Click "Upload images"
5. In Step 2, multi-select: Bluebonnet, YFH, Cotton
6. Set epochs to 10
7. Click "Start training on selected species"

### Example 2: Strict Unknown Detection
1. Go to Identification tab
2. Set confidence threshold to 0.99 (very strict)
3. Upload plant images
4. Plants with < 99% confidence will be marked "unknown"
5. Review statistics in Database tab

### Example 3: Lenient Detection
1. Go to Identification tab
2. Set confidence threshold to 0.70 (lenient)
3. Upload plant images
4. More plants will be identified (fewer marked "unknown")

## Benefits

✅ **Scalability**: Add dozens of species and train on all simultaneously
✅ **Flexibility**: Adjust confidence threshold per detection session
✅ **Clarity**: Visual feedback for uncertain detections
✅ **Statistics**: Track identification success and unknown rates
✅ **Quality Control**: Visual warnings help identify when training data is needed

### New Data Management Features

- Training dataset can now be edited from within the app:
  - Remove individual species folders
  - Wipe all training images in one click (with confirmation)
  - Helper utilities `remove_species_data` and `wipe_training_data` exposed for scripting
- Database improvements:
  - **Image ID** (filename) stored with every detection
  - New **Is Correct** flag lets users mark records as incorrectly identified
  - Column appears in editable table so mistakes can be flagged manually
  - Tab includes a **clear all detections** button with confirmation
  - `Database.clear_detections()` added for programmatic wipes


## Backward Compatibility

- All existing models and data are compatible
- No database migration needed
- Existing species labels preserved
- Can continue using single-species training if preferred

## Notes

- Lower thresholds may result in more "unknown" classifications (good when unsure)
- Higher thresholds may incorrectly classify plants (bad when few trained species)
- For best results: train on diverse images of each species (50+ images recommended)
- Unknown plants can be manually corrected in the Database tab
