// random_harness.cc: prints std::mt19937 words and libc++'s <random> draws for fixed seeds,
// through the same calls colmap/math/random.h makes. Built and run by
// oracle/fixture_random.py, which turns the output into
// colmap-rust/tests/data/oracle/random.json. Not part of any build.
//
// Why a harness and not pycolmap: pycolmap 4.2.0 binds only set_random_seed, and nothing it
// exposes returns raw draws. COLMAP's RandomUniformInteger/Real/Gaussian are header
// templates, so their numbers are exactly those of <random> in the C++ standard library
// the wheel was compiled against - libc++ on macOS (oracle/probe_random.py). Compiling this
// file with Apple clang against the same libc++ reproduces them.
//
// The Random* helpers below restate colmap/math/random.h's templates (BSD-3, COLMAP's own
// code): a fresh distribution per call on a thread-local mt19937. Output: one line per case,
// "<name> <kind> <values...>", kind u = unsigned decimal, i = signed decimal, f = C99 hex
// float (%a, exact).

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

std::unique_ptr<std::mt19937> PRNG;

void SetPRNGSeed(unsigned seed) { PRNG = std::make_unique<std::mt19937>(seed); }

template <typename T>
T RandomUniformInteger(T min, T max) {
  std::uniform_int_distribution<T> distribution(min, max);
  return distribution(*PRNG);
}

template <typename T>
T RandomUniformReal(T min, T max) {
  std::uniform_real_distribution<T> distribution(min, max);
  return distribution(*PRNG);
}

template <typename T>
T RandomGaussian(T mean, T stddev) {
  std::normal_distribution<T> distribution(mean, stddev);
  return distribution(*PRNG);
}

template <typename T>
void Shuffle(uint32_t num_to_shuffle, std::vector<T>* elems) {
  const uint32_t last_idx = static_cast<uint32_t>(elems->size() - 1);
  for (uint32_t i = 0; i < num_to_shuffle; ++i) {
    const auto j = RandomUniformInteger<uint32_t>(i, last_idx);
    std::swap((*elems)[i], (*elems)[j]);
  }
}

void Print(const std::string& name, const std::vector<uint64_t>& values) {
  std::printf("%s u", name.c_str());
  for (uint64_t v : values) std::printf(" %" PRIu64, v);
  std::printf("\n");
}

void Print(const std::string& name, const std::vector<int64_t>& values) {
  std::printf("%s i", name.c_str());
  for (int64_t v : values) std::printf(" %" PRId64, v);
  std::printf("\n");
}

void Print(const std::string& name, const std::vector<double>& values) {
  std::printf("%s f", name.c_str());
  for (double v : values) std::printf(" %a", v);
  std::printf("\n");
}

template <typename T, typename Draw>
void Case(const std::string& name, unsigned seed, int count, Draw draw) {
  SetPRNGSeed(seed);
  std::vector<T> values;
  for (int i = 0; i < count; ++i) values.push_back(static_cast<T>(draw()));
  Print(name + "/seed" + std::to_string(seed), values);
}

}  // namespace

int main() {
  {
    std::mt19937 engine;  // default_seed 5489
    for (int i = 1; i < 10000; ++i) engine();
    Print("mt19937_default_10000th", std::vector<uint64_t>{engine()});
  }

  for (unsigned seed : {0u, 1u, 42u, 4294967295u}) {
    Case<uint64_t>("mt19937", seed, 32, [] { return (*PRNG)(); });
    Case<int64_t>("int_0_10000", seed, 64, [] { return RandomUniformInteger<int>(0, 10000); });
    Case<int64_t>("int_m100_100", seed, 64, [] { return RandomUniformInteger<int>(-100, 100); });
    Case<int64_t>("int_full", seed, 16, [] {
      return RandomUniformInteger<int>(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    });
    Case<int64_t>("short_m7_300", seed, 32, [] { return RandomUniformInteger<short>(-7, 300); });
    Case<uint64_t>("uint32_0_999", seed, 64, [] { return RandomUniformInteger<uint32_t>(0, 999); });
    Case<int64_t>("int64_pm1e12", seed, 32, [] {
      return RandomUniformInteger<int64_t>(-1000000000000LL, 1000000000000LL);
    });
    Case<uint64_t>("uint64_0_2pow40p3", seed, 32, [] {
      return RandomUniformInteger<uint64_t>(0, (uint64_t(1) << 40) + 3);
    });
    Case<uint64_t>("uint64_full", seed, 16, [] {
      return RandomUniformInteger<uint64_t>(0, std::numeric_limits<uint64_t>::max());
    });
    Case<uint64_t>("size_t_0_5", seed, 32, [] { return RandomUniformInteger<size_t>(0, 5); });
    Case<double>("double_m100_100", seed, 32, [] { return RandomUniformReal<double>(-100, 100); });
    Case<double>("double_0_1", seed, 32, [] { return RandomUniformReal<double>(0, 1); });
    Case<double>("float_m1_1", seed, 32, [] { return RandomUniformReal<float>(-1, 1); });
    Case<double>("float_0_1000", seed, 32, [] { return RandomUniformReal<float>(0, 1000); });
    Case<double>("gaussian_double_1_1", seed, 32, [] { return RandomGaussian<double>(1, 1); });
    Case<double>("gaussian_float_0_2", seed, 32, [] { return RandomGaussian<float>(0, 2); });
    {
      // One distribution reused: exercises libc++'s cached second polar value.
      SetPRNGSeed(seed);
      std::normal_distribution<double> distribution(0.5, 3.0);
      std::vector<double> values;
      for (int i = 0; i < 32; ++i) values.push_back(distribution(*PRNG));
      Print("normal_reused_0.5_3/seed" + std::to_string(seed), values);
    }
    for (uint32_t num : {5u, 20u}) {
      SetPRNGSeed(seed);
      std::vector<int64_t> values(20);
      std::iota(values.begin(), values.end(), 0);
      Shuffle(num, &values);
      Print("colmap_shuffle_" + std::to_string(num) + "_of_20/seed" + std::to_string(seed), values);
    }
    {
      SetPRNGSeed(seed);
      std::vector<int64_t> values(30);
      std::iota(values.begin(), values.end(), 0);
      std::shuffle(values.begin(), values.end(), *PRNG);
      Print("std_shuffle_30/seed" + std::to_string(seed), values);
    }
  }
  return 0;
}
