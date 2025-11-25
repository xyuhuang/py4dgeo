import py4dgeo
import numpy as np
import pytest
import os
import logging
import pandas as pd

def test_read_segmented_epochs(epochs_segmented, pbm3c2_correspondences_file):
    epoch0, epoch1 = epochs_segmented
    correspondences_file = pbm3c2_correspondences_file

    assert epoch0 is not None
    assert epoch1 is not None
    assert os.path.exists(correspondences_file), f"Correspondences file should exist: {correspondences_file}"
    
    assert epoch0.cloud.shape[0] > 0, "epoch0 should have points"
    assert epoch1.cloud.shape[0] > 0, "epoch1 should have points"

    assert epoch0.additional_dimensions is not None, "epoch0 additional_dimensions should not be None"
    assert epoch1.additional_dimensions is not None, "epoch1 additional_dimensions should not be None"


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup logging handlers after test to prevent Windows file lock issues"""
    yield
    for logger_name in ['py4dgeo', 'root']:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers[:]:
            try:
                handler.close()
                logger.removeHandler(handler)
            except:
                pass

# def test_leak_on_reading_both_segment_ids(epochs_segmented):
#     print("--- Running precise test: Accessing segment_id from BOTH epochs ---")
    
#     epoch0, epoch1 = epochs_segmented

#     print("Step 1: Accessing epoch0.additional_dimensions['segment_id']")
#     try:
#         ids0 = np.unique(epoch0.additional_dimensions["segment_id"])
#         print(f"--- Successfully read {len(ids0)} unique IDs from epoch0 ---")
#     except Exception as e:
#         pytest.fail(f"Test failed at epoch0 access: {e}")

#     print("Step 2: Accessing epoch1.additional_dimensions['segment_id']")
#     try:
#         ids1 = np.unique(epoch1.additional_dimensions["segment_id"])
#         print(f"--- Successfully read {len(ids1)} unique IDs from epoch1 ---")
#     except Exception as e:
#         pytest.fail(f"Test failed at epoch1 access: {e}")

#     print("--- Precise test completed without Python exceptions. Awaiting LeakSanitizer result. ---")


# def test_preprocess(epochs_segmented,pbm3c2_correspondences_file):
#     epoch0, epoch1 = epochs_segmented
#     correspondences_file = pbm3c2_correspondences_file

#     alg = py4dgeo.PBM3C2(registration_error=0.01)
#     epoch0_preprocessed, epoch1_preprocessed, correspondences_df =alg.preprocess_epochs(epoch0, epoch1, correspondences_file)

#     assert epoch0_preprocessed is not None
#     assert epoch1_preprocessed is not None
#     assert correspondences_df is not None

#     assert epoch0_preprocessed.cloud.shape[0] > 0, "Preprocessed epoch0 should have points"
#     assert epoch1_preprocessed.cloud.shape[0] > 0, "Preprocessed epoch1 should have points"



def test_knockout_stage_1_read_segment_ids(epochs_segmented):
    """STAGE 1: Re-confirm that reading segment IDs is safe."""
    print("\n--- KNOCKOUT STAGE 1: Reading segment IDs ---")
    epoch0, epoch1 = epochs_segmented
    ids0 = np.unique(epoch0.additional_dimensions["segment_id"])
    ids1 = np.unique(epoch1.additional_dimensions["segment_id"])
    print("-> STAGE 1 PASSED (in Python)")

def test_knockout_stage_2_read_csv(epochs_segmented, pbm3c2_correspondences_file):
    """STAGE 2: Adds the pd.read_csv call."""
    print("\n--- KNOCKOUT STAGE 2: Reading CSV file ---")
    epoch0, epoch1 = epochs_segmented
    ids0 = np.unique(epoch0.additional_dimensions["segment_id"])
    ids1 = np.unique(epoch1.additional_dimensions["segment_id"])

    # Line under test:
    correspondences_df = pd.read_csv(pbm3c2_correspondences_file, header=None)
    
    print("-> STAGE 2 PASSED (in Python)")

# def test_knockout_stage_3_if_condition(epochs_segmented, pbm3c2_correspondences_file):
#     """STAGE 3: Adds the 'if' condition check."""
#     print("\n--- KNOCKOUT STAGE 3: Evaluating 'if' condition ---")
#     epoch0, epoch1 = epochs_segmented
#     ids0 = np.unique(epoch0.additional_dimensions["segment_id"])
#     ids1 = np.unique(epoch1.additional_dimensions["segment_id"])
#     correspondences_df = pd.read_csv(pbm3c2_correspondences_file, header=None)

#     # Line under test:
#     condition = not set(ids0).isdisjoint(set(ids1))
    
#     print(f"-> STAGE 3 PASSED (in Python). Condition is {condition}")

# def test_knockout_stage_4_direct_modification(epochs_segmented, pbm3c2_correspondences_file):
#     """STAGE 4: Enters the 'if' block and tests direct modification."""
#     print("\n--- KNOCKOUT STAGE 4: Direct modification of attribute ---")
#     epoch0, epoch1 = epochs_segmented
#     ids0 = np.unique(epoch0.additional_dimensions["segment_id"])
#     ids1 = np.unique(epoch1.additional_dimensions["segment_id"])
#     correspondences_df = pd.read_csv(pbm3c2_correspondences_file, header=None)

#     if not set(ids0).isdisjoint(set(ids1)):
#         max_id_epoch0 = ids0.max()
#         offset = max_id_epoch0 + 1

#         # Line under test: This is one of the original failure modes
#         epoch1.additional_dimensions["segment_id"] = epoch1.additional_dimensions["segment_id"] + offset

#     print("-> STAGE 4 PASSED (in Python)")

# def test_knockout_stage_5_epoch_recreation(epochs_segmented, pbm3c2_correspondences_file):
#     """STAGE 5: Enters the 'if' block and tests Epoch recreation."""
#     print("\n--- KNOCKOUT STAGE 5: Epoch re-creation ---")
#     epoch0, epoch1 = epochs_segmented
#     ids0 = np.unique(epoch0.additional_dimensions["segment_id"])
#     ids1 = np.unique(epoch1.additional_dimensions["segment_id"])
#     correspondences_df = pd.read_csv(pbm3c2_correspondences_file, header=None)

#     if not set(ids0).isdisjoint(set(ids1)):
#         max_id_epoch0 = ids0.max()
#         offset = max_id_epoch0 + 1

#         new_add_dims = epoch1.additional_dimensions.copy()
#         new_add_dims["segment_id"] = new_add_dims["segment_id"] + offset

#         # Line under test: This is the other failure mode
#         new_epoch1 = py4dgeo.Epoch(cloud=epoch1.cloud, additional_dimensions=new_add_dims)

#     print("-> STAGE 5 PASSED (in Python)")

# def test_compute_distances(epochs_segmented, pbm3c2_correspondences_file):
#     epoch0, epoch1 = epochs_segmented
#     correspondences_file = pbm3c2_correspondences_file
#     apply_ids = np.arange(1, 31)

#     alg = py4dgeo.PBM3C2(registration_error=0.01)

#     rez = alg.run(
#         epoch0=epoch0,
#         epoch1=epoch1,
#         correspondences_file=correspondences_file,
#         apply_ids=apply_ids,
#         search_radius=5.0,
#     )

#     assert rez is not None