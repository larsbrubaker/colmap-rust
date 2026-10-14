//! Port of the pieces of VLFeat's `mathop.h` that SIFT uses: the `VL_PI` / `VL_EPSILON_*`
//! constants, `vl_mod_2pi_f`, `vl_floor_f` / `_d`, `vl_fast_atan2_f`, `vl_fast_resqrt_f` and
//! `vl_fast_sqrt_f`, plus `sift.c`'s private `fast_expn` table. Port of colmap-sharp's
//! `Feature/VLFeat/VlMathOp.cs`. Neighbors: [`super::imopv`] (the column convolution) and the
//! SIFT filter ([`super::sift_filter`] and its detect / descriptor parts) that call these.
//!
//! Tier A (exact). These are VLFeat's own approximations, computed in the C's precision (float
//! stays float; the magic-constant inverse square root reinterprets the same bits). Mixed
//! float/double expressions keep the C promotion points, spelled out at the call sites.
//! COLMAP builds VLFeat without SSE2 on arm64 (`thirdparty/VLFeat/CMakeLists.txt`), so the
//! scalar code here is the code the macOS wheel runs. The `fast_expn` table is a constant of
//! correctly rounded values rather than `exp` calls (docs/CPP_DIVERGENCES.md entry 202).

/// `VL_PI` (`3.141592653589793`, the same double as `std::f64::consts::PI`).
pub const VL_PI: f64 = std::f64::consts::PI;

/// `VL_LOG_OF_2`: VLFeat's truncated literal, which is not the double nearest ln 2
/// (`std::f64::consts::LN_2`), so the octave count matches VLFeat's.
#[allow(clippy::approx_constant)]
pub const VL_LOG_OF_2: f64 = 0.693147180559945;

/// `VL_EPSILON_F` (2^-23).
pub const VL_EPSILON_F: f32 = 1.192_092_9E-7;

/// `VL_EPSILON_D` (2^-52).
pub const VL_EPSILON_D: f64 = 2.220446049250313e-16;

/// Port of `vl_mod_2pi_f`: `x` modulo 2*pi for arguments near [0, 2*pi).
pub fn vl_mod_2pi_f(mut x: f32) -> f32 {
    let two_pi = (2.0 * VL_PI) as f32;
    while x > two_pi {
        x -= two_pi;
    }
    while x < 0.0f32 {
        x += two_pi;
    }
    x
}

/// Port of `vl_floor_f`: floor, then convert to integer. (`(long) x` truncates; `as`
/// saturates where C is undefined, out of the `i64` range or NaN.)
pub fn vl_floor_f(x: f32) -> i64 {
    let xi = x as i64;
    if x >= 0.0 || xi as f32 == x {
        xi
    } else {
        xi - 1
    }
}

/// Port of `vl_floor_d`.
pub fn vl_floor_d(x: f64) -> i64 {
    let xi = x as i64;
    if x >= 0.0 || xi as f64 == x {
        xi
    } else {
        xi - 1
    }
}

/// Port of `vl_fast_atan2_f`: a cubic approximation of atan2 (max error about 0.004 rad).
pub fn vl_fast_atan2_f(y: f32, x: f32) -> f32 {
    const C3: f32 = 0.1821;
    const C1: f32 = 0.9675;
    let abs_y = y.abs() + VL_EPSILON_F;
    let (r, mut angle) = if x >= 0.0 {
        ((x - abs_y) / (x + abs_y), (VL_PI / 4.0) as f32)
    } else {
        ((x + abs_y) / (abs_y - x), (3.0 * VL_PI / 4.0) as f32)
    };
    angle += (C3 * r * r - C1) * r;
    if y < 0.0 {
        -angle
    } else {
        angle
    }
}

/// Port of `vl_fast_resqrt_f`: the 0x5f3759df inverse square root with two Newton steps.
pub fn vl_fast_resqrt_f(x: f32) -> f32 {
    let xhalf = 0.5f32 * x;
    let mut i = x.to_bits() as i32;
    i = 0x5f37_59df_i32.wrapping_sub(i >> 1);
    let mut u = f32::from_bits(i as u32);
    u = u * (1.5f32 - xhalf * u * u);
    u = u * (1.5f32 - xhalf * u * u);
    u
}

/// Port of `vl_fast_sqrt_f`: `x * resqrt(x)`, zero below 1e-8.
pub fn vl_fast_sqrt_f(x: f32) -> f32 {
    // `x < 1e-8` compares the float promoted to double with the double literal.
    if f64::from(x) < 1e-8 {
        0.0
    } else {
        x * vl_fast_resqrt_f(x)
    }
}

/// `sift.c`'s `EXPN_SZ`.
pub const EXPN_SZ: usize = 256;
/// `sift.c`'s `EXPN_MAX`.
pub const EXPN_MAX: f64 = 25.0;

/// `sift.c`'s `expn_tab` as `fast_expn_init` fills it: `exp(-k * (EXPN_MAX / EXPN_SZ))` for
/// `k = 0..=256`, each value the correctly rounded double of that exponential (written as bits
/// so no parse can round it). Apple libm, which the pycolmap wheel and the VLFeat fixture run,
/// returns exactly these values for all 257 arguments; the `libm` crate's `exp` is 1 ulp off
/// for some of them, which moves SIFT orientations by an ulp. A constant table gives
/// VLFeat-on-macOS's bits on every target, native and wasm (docs/CPP_DIVERGENCES.md entry
/// 202). Generated with Python's `decimal` at 80 digits, then rounded to double;
/// `tests/feature/rust_only_vl_sift_filter.rs` checks it against [`crate::math::fns::exp`]
/// (within 1 ulp) and the fixture pins it end to end.
pub const EXPN_TAB: [f64; EXPN_SZ + 1] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3fed05d24612c2af),
    f64::from_bits(0x3fea528e2e1d9f0a),
    f64::from_bits(0x3fe7df9ab76b20fd),
    f64::from_bits(0x3fe5a6fc061433c8),
    f64::from_bits(0x3fe3a344c42322f0),
    f64::from_bits(0x3fe1cf88def26e0a),
    f64::from_bits(0x3fe0275180612d31),
    f64::from_bits(0x3fdd4d244cf4ea9e),
    f64::from_bits(0x3fda933d7dd220fc),
    f64::from_bits(0x3fd81a455c174b70),
    f64::from_bits(0x3fd5dc31577a457e),
    f64::from_bits(0x3fd3d386c3dec4f1),
    f64::from_bits(0x3fd1fb4d761f3916),
    f64::from_bits(0x3fd04f039fb35bfd),
    f64::from_bits(0x3fcd952597135e09),
    f64::from_bits(0x3fcad48bc25771c7),
    f64::from_bits(0x3fc855802b8b74a7),
    f64::from_bits(0x3fc611e969df99c8),
    f64::from_bits(0x3fc4043f5a4b1409),
    f64::from_bits(0x3fc2277d9b6eed30),
    f64::from_bits(0x3fc077174b6623be),
    f64::from_bits(0x3fbdddd7d31ec28c),
    f64::from_bits(0x3fbb167a824c0132),
    f64::from_bits(0x3fb8914c880e4177),
    f64::from_bits(0x3fb648257e94a907),
    f64::from_bits(0x3fb4356faad37142),
    f64::from_bits(0x3fb2541a572fde00),
    f64::from_bits(0x3fb09f8d73309544),
    f64::from_bits(0x3fae273cb3ea1e33),
    f64::from_bits(0x3fab590b480e26f2),
    f64::from_bits(0x3fa8cdabd74cef83),
    f64::from_bits(0x3fa67ee6d9ff847c),
    f64::from_bits(0x3fa46718dbaf5fd5),
    f64::from_bits(0x3fa28124b439e06e),
    f64::from_bits(0x3fa0c8670916d4ad),
    f64::from_bits(0x3f9e7155f0750059),
    f64::from_bits(0x3f9b9c3fa1c5598c),
    f64::from_bits(0x3f990a9f8263b0ac),
    f64::from_bits(0x3f96b62ec3a36a9e),
    f64::from_bits(0x3f94993c15e964fb),
    f64::from_bits(0x3f92ae9dbff486ba),
    f64::from_bits(0x3f90f1a5016fc0f8),
    f64::from_bits(0x3f8ebc2543f5c374),
    f64::from_bits(0x3f8be019216b7bda),
    f64::from_bits(0x3f894828f5e61af9),
    f64::from_bits(0x3f86edfe86286e1c),
    f64::from_bits(0x3f84cbda8565f8a9),
    f64::from_bits(0x3f82dc868a5d6bea),
    f64::from_bits(0x3f811b4852eaa984),
    f64::from_bits(0x3f7f07ac6de3e81e),
    f64::from_bits(0x3f7c24995cd64169),
    f64::from_bits(0x3f798649a1e7ad48),
    f64::from_bits(0x3f7726576f632fa2),
    f64::from_bits(0x3f74fef558ea8751),
    f64::from_bits(0x3f730ae0260e8ea3),
    f64::from_bits(0x3f714551f6951195),
    f64::from_bits(0x3f6f53ed32028990),
    f64::from_bits(0x3f6c69c1edc0a9eb),
    f64::from_bits(0x3f69c502fa0468ea),
    f64::from_bits(0x3f675f3ad05caa9d),
    f64::from_bits(0x3f65328dc22484e3),
    f64::from_bits(0x3f6339aba844bbb1),
    f64::from_bits(0x3f616fc2e7e08200),
    f64::from_bits(0x3f5fa0e9586aebc7),
    f64::from_bits(0x3f5caf9471d49400),
    f64::from_bits(0x3f5a04567569805c),
    f64::from_bits(0x3f5798a9fd5a152d),
    f64::from_bits(0x3f5566a4f5b0912c),
    f64::from_bits(0x3f5368ea28e60847),
    f64::from_bits(0x3f519a9c24a8692a),
    f64::from_bits(0x3f4feea2ad9723a3),
    f64::from_bits(0x3f4cf6128ab46776),
    f64::from_bits(0x3f4a44458ede1b5f),
    f64::from_bits(0x3f47d2a64de4d371),
    f64::from_bits(0x3f459b3c2b21adc4),
    f64::from_bits(0x3f43989cc2885c2d),
    f64::from_bits(0x3f41c5dead380970),
    f64::from_bits(0x3f401e8d81366c98),
    f64::from_bits(0x3f3d3d3dde04d757),
    f64::from_bits(0x3f3a84d1c4cc3094),
    f64::from_bits(0x3f380d311cd27e54),
    f64::from_bits(0x3f35d0549d0885b3),
    f64::from_bits(0x3f33c8c492780c05),
    f64::from_bits(0x3f31f18b84507626),
    f64::from_bits(0x3f30462a16241230),
    f64::from_bits(0x3f2d85181576bbea),
    f64::from_bits(0x3f2ac5fc994974d9),
    f64::from_bits(0x3f28484bc84cfe21),
    f64::from_bits(0x3f2605ef88fac6fb),
    f64::from_bits(0x3f23f962b8be83bf),
    f64::from_bits(0x3f221da3af2e9f4d),
    f64::from_bits(0x3f206e280283423a),
    f64::from_bits(0x3f1dcda2ded104ed),
    f64::from_bits(0x3f1b07c79220609f),
    f64::from_bits(0x3f1883f7b1dab8f7),
    f64::from_bits(0x3f163c0e2f9a8e2e),
    f64::from_bits(0x3f142a785829017c),
    f64::from_bits(0x3f124a2835916c22),
    f64::from_bits(0x3f1096883588f934),
    f64::from_bits(0x3f0e16dfebfac43b),
    f64::from_bits(0x3f0b4a3438d94b63),
    f64::from_bits(0x3f08c0363e66d558),
    f64::from_bits(0x3f0672b1d49de448),
    f64::from_bits(0x3f045c06964f60f0),
    f64::from_bits(0x3f02771a21bfe4b3),
    f64::from_bits(0x3f00bf4ba0b60739),
    f64::from_bits(0x3efe60d0f305512a),
    f64::from_bits(0x3efb8d441ac39d8f),
    f64::from_bits(0x3ef8fd08d649910e),
    f64::from_bits(0x3ef6a9dbbed64ee7),
    f64::from_bits(0x3ef48e0e9b9af77a),
    f64::from_bits(0x3ef2a47a808f6a91),
    f64::from_bits(0x3ef0e87337dcb492),
    f64::from_bits(0x3eeeab77ae3684c4),
    f64::from_bits(0x3eebd0f8c8ff18f3),
    f64::from_bits(0x3ee93a70e550ac81),
    f64::from_bits(0x3ee6e18d38387323),
    f64::from_bits(0x3ee4c091934d811b),
    f64::from_bits(0x3ee2d24a616a00ce),
    f64::from_bits(0x3ee111fff1267413),
    f64::from_bits(0x3edef6d5dc130f34),
    f64::from_bits(0x3edc1553d885380a),
    f64::from_bits(0x3ed9786fdac7eac7),
    f64::from_bits(0x3ed719c78de3cb24),
    f64::from_bits(0x3ed4f390ab881e69),
    f64::from_bits(0x3ed3008ad654a370),
    f64::from_bits(0x3ed13bf2c519a381),
    f64::from_bits(0x3ecf42ed3f68e690),
    f64::from_bits(0x3ecc5a56e232a448),
    f64::from_bits(0x3ec9b7072981a695),
    f64::from_bits(0x3ec7528c102a6ead),
    f64::from_bits(0x3ec5270d155263bb),
    f64::from_bits(0x3ec32f3cf3f5ae74),
    f64::from_bits(0x3ec1664cae9f5a18),
    f64::from_bits(0x3ebf8fbf9f59cf4c),
    f64::from_bits(0x3ebca00382d0c3af),
    f64::from_bits(0x3eb9f63847df7c46),
    f64::from_bits(0x3eb78bdc1298eec5),
    f64::from_bits(0x3eb55b0804a179a8),
    f64::from_bits(0x3eb35e61d19b548a),
    f64::from_bits(0x3eb1910eab094556),
    f64::from_bits(0x3eafdd4ec765fe86),
    f64::from_bits(0x3eace65b5b1f5dd3),
    f64::from_bits(0x3eaa3604afdb0929),
    f64::from_bits(0x3ea7c5b8ebfe44ae),
    f64::from_bits(0x3ea58f82b05f4f0a),
    f64::from_bits(0x3ea38dfa894225be),
    f64::from_bits(0x3ea1bc39ba17942d),
    f64::from_bits(0x3ea015ce43bb6b40),
    f64::from_bits(0x3e9d2d600fde589a),
    f64::from_bits(0x3e9a766ddf0ec058),
    f64::from_bits(0x3e980023f673d463),
    f64::from_bits(0x3e95c47e5271dcaf),
    f64::from_bits(0x3e93be08379ba61e),
    f64::from_bits(0x3e91e7ceddfef0c4),
    f64::from_bits(0x3e903d5559f4deba),
    f64::from_bits(0x3e8d751349d78ce7),
    f64::from_bits(0x3e8ab77556bed539),
    f64::from_bits(0x3e883b1e8f6582bc),
    f64::from_bits(0x3e85f9fc27c27ae6),
    f64::from_bits(0x3e83ee8bfc14f499),
    f64::from_bits(0x3e8213cf1b6e88e3),
    f64::from_bits(0x3e80653d92cd8784),
    f64::from_bits(0x3e7dbd76b5e8b36a),
    f64::from_bits(0x3e7af91c9be23beb),
    f64::from_bits(0x3e7876aa1799df72),
    f64::from_bits(0x3e762ffd70454914),
    f64::from_bits(0x3e741f86f8dd8235),
    f64::from_bits(0x3e72403b79962533),
    f64::from_bits(0x3e708d87dcf8945a),
    f64::from_bits(0x3e6e068c050d69d9),
    f64::from_bits(0x3e6b3b65372bbfd3),
    f64::from_bits(0x3e68b2c7f33a633b),
    f64::from_bits(0x3e6666836f00a773),
    f64::from_bits(0x3e6450fa52edd9c4),
    f64::from_bits(0x3e626d15022c4f78),
    f64::from_bits(0x3e60b6352973c89f),
    f64::from_bits(0x3e5e5054ec6950c7),
    f64::from_bits(0x3e5b7e50b513307e),
    f64::from_bits(0x3e58ef7989dbc21d),
    f64::from_bits(0x3e569d8f6a14c339),
    f64::from_bits(0x3e5482e7320e7842),
    f64::from_bits(0x3e529a5cc17487e7),
    f64::from_bits(0x3e50df466b8d1dd4),
    f64::from_bits(0x3e4e9ad325524257),
    f64::from_bits(0x3e4bc1e0a5dea510),
    f64::from_bits(0x3e492cc04686523d),
    f64::from_bits(0x3e46d522aac33545),
    f64::from_bits(0x3e44b54ec0deb60a),
    f64::from_bits(0x3e42c813c64589ba),
    f64::from_bits(0x3e4108bc98e872e7),
    f64::from_bits(0x3e3ee6086d5aa206),
    f64::from_bits(0x3e3c06169dabd671),
    f64::from_bits(0x3e396a9d97be874b),
    f64::from_bits(0x3e370d3e7d76b38a),
    f64::from_bits(0x3e34e8322cdbc100),
    f64::from_bits(0x3e32f63b220f9f30),
    f64::from_bits(0x3e313298a9854973),
    f64::from_bits(0x3e2f31f6865bc5d0),
    f64::from_bits(0x3e2c4af434799080),
    f64::from_bits(0x3e29a912ef8d82ce),
    f64::from_bits(0x3e2745e431cad55b),
    f64::from_bits(0x3e251b92a667a7e3),
    f64::from_bits(0x3e2324d3e8e3051a),
    f64::from_bits(0x3e215cdb97c49126),
    f64::from_bits(0x3e1f7e9f368078e9),
    f64::from_bits(0x3e1c907b06313a6e),
    f64::from_bits(0x3e19e821c389b973),
    f64::from_bits(0x3e177f151a93ead8),
    f64::from_bits(0x3e154f7160d076f4),
    f64::from_bits(0x3e1353df31765e12),
    f64::from_bits(0x3e118786606e8158),
    f64::from_bits(0x3e0fcc04484f9849),
    f64::from_bits(0x3e0cd6acb2b07690),
    f64::from_bits(0x3e0a27cb8cdfada6),
    f64::from_bits(0x3e07b8d28de6e794),
    f64::from_bits(0x3e0583cf9257660d),
    f64::from_bits(0x3e03835e152d359b),
    f64::from_bits(0x3e01b29a02b88108),
    f64::from_bits(0x3e000d13c55b64a6),
    f64::from_bits(0x3dfd1d8addd2d9d0),
    f64::from_bits(0x3dfa6811c85abf93),
    f64::from_bits(0x3df7f31de52160b7),
    f64::from_bits(0x3df5b8ae74381863),
    f64::from_bits(0x3df3b351b01e9334),
    f64::from_bits(0x3df1de17804b1d61),
    f64::from_bits(0x3df03485688aa553),
    f64::from_bits(0x3ded65172f7bbb04),
    f64::from_bits(0x3deaa8f5f66e12d7),
    f64::from_bits(0x3de82df87cf19ebf),
    f64::from_bits(0x3de5ee0f42afee18),
    f64::from_bits(0x3de3e3bb211b9d9b),
    f64::from_bits(0x3de209ffdd480edd),
    f64::from_bits(0x3de05c57f9a36ff7),
    f64::from_bits(0x3dddad5353a01a5d),
    f64::from_bits(0x3ddaea799b3d8a06),
    f64::from_bits(0x3dd86963b55ec320),
    f64::from_bits(0x3dd623f33d0567ca),
    f64::from_bits(0x3dd4149b89b64e5c),
    f64::from_bits(0x3dd2365420504d3c),
    f64::from_bits(0x3dd0848c66d76bfc),
    f64::from_bits(0x3dcdf640fa50a132),
    f64::from_bits(0x3dcb2c9e3ea6d850),
    f64::from_bits(0x3dc8a560f1d101d6),
    f64::from_bits(0x3dc65a5ba58f9c47),
    f64::from_bits(0x3dc445f40e4835db),
    f64::from_bits(0x3dc26315528a3260),
    f64::from_bits(0x3dc0ad23a0a19728),
    f64::from_bits(0x3dbe3fe1d7c3ba50),
    f64::from_bits(0x3dbb6f656c4aa962),
    f64::from_bits(0x3db8e1f19919ef44),
    f64::from_bits(0x3db69149c1bdc096),
    f64::from_bits(0x3db477c5d5f94ff9),
    f64::from_bits(0x3db290447fa7ac4c),
    f64::from_bits(0x3db0d61e99cbe402),
    f64::from_bits(0x3dae8a37a45fc32e),
];

/// Port of `sift.c`'s `fast_expn`: `exp(-x)` by linear interpolation in the table; zero
/// beyond `EXPN_MAX`. (VLFeat asserts `0 <= x`; every caller passes a non-negative ratio.)
pub fn fast_expn(mut x: f64) -> f64 {
    if x > EXPN_MAX {
        return 0.0;
    }
    x *= EXPN_SZ as f64 / EXPN_MAX;
    let i = vl_floor_d(x) as usize;
    let r = x - i as f64;
    let a = EXPN_TAB[i];
    let b = EXPN_TAB[i + 1];
    a + r * (b - a)
}
