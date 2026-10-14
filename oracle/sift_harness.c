// sift_harness.c: runs VLFeat's SIFT (cpp-reference/src/thirdparty/VLFeat, BSD-2) through the
// same calls COLMAP's SiftCPUFeatureExtractor makes and prints every keypoint, orientation and
// raw float descriptor exactly (C99 hex floats). Built and run by oracle/fixture_sift.py, which
// turns the output into colmap-rust/tests/data/oracle/sift_vlfeat.json. Not part of any build.
// Ported from colmap-sharp's oracle/sift_harness.c (unchanged apart from these comments); its
// covdet mode is for colmap-sharp's fixture_covariant_sift.py, not ported yet.
//
// Why a harness in addition to pycolmap: Apple clang contracts a*b + c into fused multiply-adds
// by default (-ffp-contract=on), and the macOS arm64 wheel's VLFeat is built that way, so its
// keypoints differ from an unfused evaluation in the last bits (colmap-sharp divergence 41). This
// harness is compiled with -ffp-contract=off (and without SSE2, like COLMAP's arm64 build) so it
// is the exact, unfused VLFeat the Rust port must match bit for bit.
//
// Usage: sift_harness <raw uint8 grey file> <width> <height> <o_min> <noctaves> <upright>
// Output per octave keypoint:  K o ix iy is x y s sigma
//        per used orientation: A angle d0 ... d127   (descriptor before COLMAP's normalization)
//
// Covariant mode drives VLFeat's covdet (covdet.c, scalespace.c) the way COLMAP's
// CovariantSiftCPUFeatureExtractor does, with its default options, and prints every feature
// in covdet's order (before COLMAP's sort and truncation) with the raw descriptor of each
// domain-size-pooling scale (one scale without DSP):
// Usage: sift_harness covdet <file> <width> <height> <first_octave> <affine> <upright> <dsp>
// Output per feature: F o s x y a11 a12 a21 a22 peakScore edgeScore orientationScore
//        per scale:   D d0 ... d127

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "covdet.h"
#include "imopv.h"
#include "sift.h"

static unsigned char* read_image(const char* path, int w, int h) {
  unsigned char* data = malloc((size_t)w * h);
  FILE* f = fopen(path, "rb");
  if (!f || fread(data, 1, (size_t)w * h, f) != (size_t)w * h) exit(1);
  fclose(f);
  return data;
}

static int covdet_main(int argc, char** argv) {
  if (argc != 9) {
    fprintf(stderr, "usage: sift_harness covdet file width height first_octave affine upright dsp\n");
    return 1;
  }
  int w = atoi(argv[3]), h = atoi(argv[4]), first_octave = atoi(argv[5]);
  int affine = atoi(argv[6]), upright = atoi(argv[7]), dsp = atoi(argv[8]);
  unsigned char* data = read_image(argv[2], w, h);
  float* im = malloc(sizeof(float) * w * h);
  for (int i = 0; i < w * h; ++i) im[i] = (float)data[i] / 255.0f;

  VlCovDet* covdet = vl_covdet_new(VL_COVDET_METHOD_DOG);
  vl_covdet_set_first_octave(covdet, first_octave);
  vl_covdet_set_octave_resolution(covdet, 3);
  vl_covdet_set_peak_threshold(covdet, 0.02 / 3);
  vl_covdet_set_edge_threshold(covdet, 10.0);
  vl_covdet_put_image(covdet, im, w, h);
  vl_covdet_detect(covdet, 8192);
  if (affine) vl_covdet_extract_affine_shape(covdet);
  if (!upright) vl_covdet_extract_orientations(covdet);

  /* COLMAP's descriptor loop (sift.cc), minus the pooling and normalization. */
  enum { kPatchResolution = 15, kPatchSide = 2 * kPatchResolution + 1 };
  const double kPatchRelativeExtent = 7.5;
  const double kPatchRelativeSmoothing = 1;
  const double kPatchStep = kPatchRelativeExtent / kPatchResolution;
  const double kSigma = kPatchRelativeExtent / (3.0 * (4 + 1) / 2) / kPatchStep;
  float patch[kPatchSide * kPatchSide];
  float patchXY[2 * kPatchSide * kPatchSide];
  float dsp_min_scale = 1, dsp_scale_step = 0;
  int dsp_num_scales = 1;
  if (dsp) {
    dsp_min_scale = 1.0 / 6.0;
    dsp_scale_step = (3.0 - 1.0 / 6.0) / 10;
    dsp_num_scales = 10;
  }
  VlSiftFilt* sift = vl_sift_new(16, 16, 1, 3, 0);
  vl_sift_set_magnif(sift, 3.0);
  float desc[128];

  int n = vl_covdet_get_num_features(covdet);
  VlCovDetFeature* features = vl_covdet_get_features(covdet);
  for (int i = 0; i < n; ++i) {
    VlCovDetFeature* f = &features[i];
    printf("F %d %d %a %a %a %a %a %a %a %a %a\n", f->o, f->s, f->frame.x, f->frame.y, f->frame.a11,
           f->frame.a12, f->frame.a21, f->frame.a22, f->peakScore, f->edgeScore, f->orientationScore);
    for (int s = 0; s < dsp_num_scales; ++s) {
      const double dsp_scale = dsp_min_scale + s * dsp_scale_step;
      VlFrameOrientedEllipse scaled_frame = f->frame;
      scaled_frame.a11 *= dsp_scale;
      scaled_frame.a12 *= dsp_scale;
      scaled_frame.a21 *= dsp_scale;
      scaled_frame.a22 *= dsp_scale;
      vl_covdet_extract_patch_for_frame(covdet, patch, kPatchResolution, kPatchRelativeExtent,
                                        kPatchRelativeSmoothing, scaled_frame);
      vl_imgradient_polar_f(patchXY, patchXY + 1, 2, 2 * kPatchSide, patch, kPatchSide, kPatchSide,
                            kPatchSide);
      vl_sift_calc_raw_descriptor(sift, patchXY, desc, kPatchSide, kPatchSide, kPatchResolution,
                                  kPatchResolution, kSigma, 0);
      printf("D");
      for (int d = 0; d < 128; ++d) printf(" %a", desc[d]);
      printf("\n");
    }
  }
  vl_sift_delete(sift);
  vl_covdet_delete(covdet);
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && strcmp(argv[1], "covdet") == 0) return covdet_main(argc, argv);
  if (argc != 7) {
    fprintf(stderr, "usage: sift_harness file width height o_min noctaves upright\n");
    return 1;
  }
  int w = atoi(argv[2]), h = atoi(argv[3]), o_min = atoi(argv[4]), noctaves = atoi(argv[5]);
  int upright = atoi(argv[6]);
  unsigned char* data = malloc((size_t)w * h);
  FILE* f = fopen(argv[1], "rb");
  if (!f || fread(data, 1, (size_t)w * h, f) != (size_t)w * h) return 1;
  fclose(f);
  float* im = malloc(sizeof(float) * w * h);
  for (int i = 0; i < w * h; ++i) im[i] = (float)data[i] / 255.0f;

  VlSiftFilt* sift = vl_sift_new(w, h, noctaves, 3, o_min);
  vl_sift_set_peak_thresh(sift, 0.02 / 3);
  vl_sift_set_edge_thresh(sift, 10.0);
  int first = 1;
  float desc[128];
  while (1) {
    if (first) {
      if (vl_sift_process_first_octave(sift, im)) break;
      first = 0;
    } else if (vl_sift_process_next_octave(sift)) {
      break;
    }
    vl_sift_detect(sift);
    const VlSiftKeypoint* k = vl_sift_get_keypoints(sift);
    int n = vl_sift_get_nkeypoints(sift);
    for (int i = 0; i < n; ++i) {
      printf("K %d %d %d %d %a %a %a %a\n", k[i].o, k[i].ix, k[i].iy, k[i].is,
             k[i].x, k[i].y, k[i].s, k[i].sigma);
      double angles[4];
      int no;
      if (upright) {
        no = 1;
        angles[0] = 0.0;
      } else {
        no = vl_sift_calc_keypoint_orientations(sift, angles, &k[i]);
      }
      for (int o = 0; o < no; ++o) {
        vl_sift_calc_keypoint_descriptor(sift, desc, &k[i], angles[o]);
        printf("A %a", angles[o]);
        for (int d = 0; d < 128; ++d) printf(" %a", desc[d]);
        printf("\n");
      }
    }
  }
  vl_sift_delete(sift);
  return 0;
}
