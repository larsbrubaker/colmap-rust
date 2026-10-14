// Port of COLMAP's src/colmap/util/ply_test.cc, 1:1 (Suite_Name -> suite_name), testing
// colmap_rust::util::ply through its file (`_path`) forms, as the C++ tests use paths. Tier A
// (exact). `CreateTestDir()` is a fresh temp directory per call.

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use colmap_rust::util::ply::*;

fn create_test_dir() -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "colmap_rust_ply_{}_{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn f32s(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

#[test]
fn ply_point_default_constructor() {
    let point = PlyPoint::default();
    assert_eq!(point.x, 0.0f32);
    assert_eq!(point.y, 0.0f32);
    assert_eq!(point.z, 0.0f32);
    assert_eq!(point.nx, 0.0f32);
    assert_eq!(point.ny, 0.0f32);
    assert_eq!(point.nz, 0.0f32);
    assert_eq!(point.r, 0);
    assert_eq!(point.g, 0);
    assert_eq!(point.b, 0);
}

#[test]
fn ply_mesh_vertex_default_constructor() {
    let vertex = PlyMeshVertex::default();
    assert_eq!(vertex.x, 0.0f32);
    assert_eq!(vertex.y, 0.0f32);
    assert_eq!(vertex.z, 0.0f32);
    assert_eq!(vertex.r, 200);
    assert_eq!(vertex.g, 200);
    assert_eq!(vertex.b, 200);
}

#[test]
fn ply_mesh_vertex_parameterized_constructor() {
    let vertex = PlyMeshVertex::new(1.5, 2.5, 3.5);
    assert_eq!(vertex.x, 1.5f32);
    assert_eq!(vertex.y, 2.5f32);
    assert_eq!(vertex.z, 3.5f32);
    assert_eq!(vertex.r, 200);
    assert_eq!(vertex.g, 200);
    assert_eq!(vertex.b, 200);
}

#[test]
fn ply_mesh_vertex_color_constructor() {
    let vertex = PlyMeshVertex::with_color(1.0, 2.0, 3.0, 10, 20, 30);
    assert_eq!(vertex.x, 1.0f32);
    assert_eq!(vertex.y, 2.0f32);
    assert_eq!(vertex.z, 3.0f32);
    assert_eq!(vertex.r, 10);
    assert_eq!(vertex.g, 20);
    assert_eq!(vertex.b, 30);
}

#[test]
fn ply_mesh_face_default_constructor() {
    let face = PlyMeshFace::default();
    assert_eq!(face.vertex_idx1, 0);
    assert_eq!(face.vertex_idx2, 0);
    assert_eq!(face.vertex_idx3, 0);
}

#[test]
fn ply_mesh_face_parameterized_constructor() {
    let face = PlyMeshFace::new(10, 20, 30);
    assert_eq!(face.vertex_idx1, 10);
    assert_eq!(face.vertex_idx2, 20);
    assert_eq!(face.vertex_idx3, 30);
}

fn assert_points_eq(loaded: &[PlyPoint], original: &[PlyPoint]) {
    assert_eq!(loaded.len(), original.len());
    for (l, o) in loaded.iter().zip(original) {
        assert_eq!(l.x, o.x);
        assert_eq!(l.y, o.y);
        assert_eq!(l.z, o.z);
        assert_eq!(l.nx, o.nx);
        assert_eq!(l.ny, o.ny);
        assert_eq!(l.nz, o.nz);
        assert_eq!(l.r, o.r);
        assert_eq!(l.g, o.g);
        assert_eq!(l.b, o.b);
    }
}

/// The single point the XYZ-only tests write, with normals and colors set.
fn xyz_only_point(x: f32, y: f32, z: f32) -> PlyPoint {
    PlyPoint {
        x,
        y,
        z,
        nx: 0.15,
        ny: 0.25,
        nz: 0.35,
        r: 15,
        g: 25,
        b: 35,
    }
}

/// Normals and colors should be default (0).
fn assert_xyz_only(loaded: &[PlyPoint], x: f32, y: f32, z: f32) {
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].x, x);
    assert_eq!(loaded[0].y, y);
    assert_eq!(loaded[0].z, z);
    assert_eq!(loaded[0].nx, 0.0f32);
    assert_eq!(loaded[0].ny, 0.0f32);
    assert_eq!(loaded[0].nz, 0.0f32);
    assert_eq!(loaded[0].r, 0);
    assert_eq!(loaded[0].g, 0);
    assert_eq!(loaded[0].b, 0);
}

#[test]
fn ply_round_trip_text_ply_points_full_data() {
    let test_file = create_test_dir().join("test.ply");
    // Create test points with full data.
    let original_points: Vec<PlyPoint> = (0..3)
        .map(|i| PlyPoint {
            x: i as f32 * 1.0f32,
            y: i as f32 * 2.0f32,
            z: i as f32 * 3.0f32,
            nx: i as f32 * 0.1f32,
            ny: i as f32 * 0.2f32,
            nz: i as f32 * 0.3f32,
            r: (i * 10) as u8,
            g: (i * 20) as u8,
            b: (i * 30) as u8,
        })
        .collect();
    write_text_ply_points_path(&test_file, &original_points, true, true).unwrap();
    let loaded_points = read_ply_path(&test_file).unwrap();
    assert_points_eq(&loaded_points, &original_points);
}

#[test]
fn ply_round_trip_text_ply_points_xyz_only() {
    let test_file = create_test_dir().join("test.ply");
    let original_points = vec![xyz_only_point(1.5, 2.5, 3.5)];
    write_text_ply_points_path(&test_file, &original_points, false, false).unwrap();
    let loaded_points = read_ply_path(&test_file).unwrap();
    assert_xyz_only(&loaded_points, 1.5, 2.5, 3.5);
}

#[test]
fn ply_round_trip_binary_ply_points_full_data() {
    let test_file = create_test_dir().join("test.ply");
    let original_points: Vec<PlyPoint> = (0..5)
        .map(|i| PlyPoint {
            x: i as f32 * 1.5f32,
            y: i as f32 * 2.5f32,
            z: i as f32 * 3.5f32,
            nx: i as f32 * 0.15f32,
            ny: i as f32 * 0.25f32,
            nz: i as f32 * 0.35f32,
            r: (i * 15) as u8,
            g: (i * 25) as u8,
            b: (i * 35) as u8,
        })
        .collect();
    write_binary_ply_points_path(&test_file, &original_points, true, true).unwrap();
    let loaded_points = read_ply_path(&test_file).unwrap();
    assert_points_eq(&loaded_points, &original_points);
}

#[test]
fn ply_round_trip_binary_ply_points_xyz_only() {
    let test_file = create_test_dir().join("test.ply");
    let original_points = vec![xyz_only_point(10.5, 20.5, 30.5)];
    write_binary_ply_points_path(&test_file, &original_points, false, false).unwrap();
    let loaded_points = read_ply_path(&test_file).unwrap();
    assert_xyz_only(&loaded_points, 10.5, 20.5, 30.5);
}

fn quad_mesh() -> PlyMesh {
    PlyMesh {
        vertices: vec![
            PlyMeshVertex::new(0.0, 0.0, 0.0),
            PlyMeshVertex::new(1.0, 0.0, 0.0),
            PlyMeshVertex::new(0.0, 1.0, 0.0),
            PlyMeshVertex::new(1.0, 1.0, 1.0),
        ],
        faces: vec![PlyMeshFace::new(0, 1, 2), PlyMeshFace::new(1, 3, 2)],
    }
}

fn assert_plain_mesh_round_trip(loaded_mesh: &PlyMesh, original_mesh: &PlyMesh) {
    assert_eq!(loaded_mesh.vertices.len(), original_mesh.vertices.len());
    assert_eq!(loaded_mesh.faces.len(), original_mesh.faces.len());
    for (l, o) in loaded_mesh.vertices.iter().zip(&original_mesh.vertices) {
        assert_eq!(l.x, o.x);
        assert_eq!(l.y, o.y);
        assert_eq!(l.z, o.z);
        // No colors in file, should get defaults.
        assert_eq!(l.r, 200);
        assert_eq!(l.g, 200);
        assert_eq!(l.b, 200);
    }
    for (l, o) in loaded_mesh.faces.iter().zip(&original_mesh.faces) {
        assert_eq!(l.vertex_idx1, o.vertex_idx1);
        assert_eq!(l.vertex_idx2, o.vertex_idx2);
        assert_eq!(l.vertex_idx3, o.vertex_idx3);
    }
}

#[test]
fn ply_round_trip_text_ply_mesh() {
    let test_file = create_test_dir().join("mesh.ply");
    let original_mesh = quad_mesh();
    write_text_ply_mesh_path(&test_file, &PlyTexturedMesh::from(original_mesh.clone())).unwrap();
    let loaded_mesh = read_ply_mesh_path(&test_file).unwrap().mesh;
    assert_plain_mesh_round_trip(&loaded_mesh, &original_mesh);
}

#[test]
fn ply_round_trip_binary_ply_mesh() {
    let test_file = create_test_dir().join("mesh.ply");
    let original_mesh = quad_mesh();
    write_binary_ply_mesh_path(&test_file, &PlyTexturedMesh::from(original_mesh.clone())).unwrap();
    let loaded_mesh = read_ply_mesh_path(&test_file).unwrap().mesh;
    assert_plain_mesh_round_trip(&loaded_mesh, &original_mesh);
}

fn create_test_textured_mesh() -> PlyTexturedMesh {
    PlyTexturedMesh {
        mesh: quad_mesh(),
        // 6 UV floats per face (u1,v1, u2,v2, u3,v3): face 0, then face 1.
        face_uvs: vec![
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, //
            0.5, 0.5, 1.0, 1.0, 0.25, 0.75,
        ],
        texture_file: "texture.png".to_string(),
    }
}

fn verify_textured_mesh(loaded: &PlyTexturedMesh, original: &PlyTexturedMesh) {
    assert_eq!(loaded.texture_file, original.texture_file);
    assert_eq!(loaded.mesh.vertices.len(), original.mesh.vertices.len());
    assert_eq!(loaded.mesh.faces.len(), original.mesh.faces.len());
    assert_eq!(loaded.face_uvs.len(), original.face_uvs.len());
    for (l, o) in loaded.mesh.vertices.iter().zip(&original.mesh.vertices) {
        assert_eq!(l.x, o.x);
        assert_eq!(l.y, o.y);
        assert_eq!(l.z, o.z);
    }
    for (l, o) in loaded.mesh.faces.iter().zip(&original.mesh.faces) {
        assert_eq!(l.vertex_idx1, o.vertex_idx1);
        assert_eq!(l.vertex_idx2, o.vertex_idx2);
        assert_eq!(l.vertex_idx3, o.vertex_idx3);
    }
    for (l, o) in loaded.face_uvs.iter().zip(&original.face_uvs) {
        assert_eq!(l, o);
    }
}

#[test]
fn ply_round_trip_text_textured_ply_mesh() {
    let test_file = create_test_dir().join("textured_mesh.ply");
    let original = create_test_textured_mesh();
    write_text_ply_mesh_path(&test_file, &original).unwrap();
    let loaded = read_ply_mesh_path(&test_file).unwrap();
    verify_textured_mesh(&loaded, &original);
}

#[test]
fn ply_round_trip_binary_textured_ply_mesh() {
    let test_file = create_test_dir().join("textured_mesh.ply");
    let original = create_test_textured_mesh();
    write_binary_ply_mesh_path(&test_file, &original).unwrap();
    let loaded = read_ply_mesh_path(&test_file).unwrap();
    verify_textured_mesh(&loaded, &original);
}

fn triangle_mesh() -> PlyMesh {
    PlyMesh {
        vertices: vec![
            PlyMeshVertex::new(0.0, 0.0, 0.0),
            PlyMeshVertex::new(1.0, 0.0, 0.0),
            PlyMeshVertex::new(0.0, 1.0, 0.0),
        ],
        faces: vec![PlyMeshFace::new(0, 1, 2)],
    }
}

#[test]
fn ply_read_ply_mesh_without_texcoords() {
    let test_file = create_test_dir().join("mesh.ply");
    // Write a plain mesh (no texcoords) and read it with ReadPlyMesh.
    write_text_ply_mesh_path(&test_file, &PlyTexturedMesh::from(triangle_mesh())).unwrap();
    let loaded = read_ply_mesh_path(&test_file).unwrap();
    assert_eq!(loaded.mesh.vertices.len(), 3);
    assert_eq!(loaded.mesh.faces.len(), 1);
    assert!(loaded.face_uvs.is_empty());
    assert!(loaded.texture_file.is_empty());
}

const COLOR_HEADER: &str = "property float x\nproperty float y\nproperty float z\n";
const RGB_PROPS: &str = "property uchar red\nproperty uchar green\nproperty uchar blue\n";
const NORMAL_PROPS: &str = "property float nx\nproperty float ny\nproperty float nz\n";
const FACE_PROPS: &str = "element face 1\nproperty list uchar int vertex_index\nend_header\n";

#[test]
fn ply_read_text_ply_mesh_with_vertex_colors() {
    let test_file = create_test_dir().join("mesh_colors.ply");
    // Write a PLY mesh with per-vertex colors manually.
    let text = format!(
        "ply\nformat ascii 1.0\nelement vertex 3\n{COLOR_HEADER}{RGB_PROPS}{FACE_PROPS}\
         0 0 0 255 0 0\n1 0 0 0 255 0\n0 1 0 0 0 255\n3 0 1 2\n"
    );
    std::fs::write(&test_file, text).unwrap();

    let mesh = read_ply_mesh_path(&test_file).unwrap().mesh;
    assert_eq!(mesh.vertices.len(), 3);
    assert_eq!(mesh.faces.len(), 1);
    let rgb = |i: usize| (mesh.vertices[i].r, mesh.vertices[i].g, mesh.vertices[i].b);
    assert_eq!(rgb(0), (255, 0, 0));
    assert_eq!(rgb(1), (0, 255, 0));
    assert_eq!(rgb(2), (0, 0, 255));
    assert_eq!(mesh.faces[0].vertex_idx1, 0);
    assert_eq!(mesh.faces[0].vertex_idx2, 1);
    assert_eq!(mesh.faces[0].vertex_idx3, 2);
}

/// The binary body of the hand-written meshes: per vertex the floats then the colors, then
/// one face `3 0 1 2` (uchar count, int indices), in native byte order as the C++ test
/// writes it.
fn binary_mesh_file(header_props: &str, vertices: &[(&[f32], [u8; 3])]) -> Vec<u8> {
    let mut bytes = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex 3\n{header_props}{FACE_PROPS}"
    )
    .into_bytes();
    for (floats, color) in vertices {
        bytes.extend(f32s(floats));
        bytes.extend_from_slice(color);
    }
    bytes.push(3);
    for idx in [0i32, 1, 2] {
        bytes.extend_from_slice(&idx.to_ne_bytes());
    }
    bytes
}

#[test]
fn ply_read_binary_ply_mesh_with_vertex_colors() {
    let test_file = create_test_dir().join("mesh_colors_bin.ply");
    let bytes = binary_mesh_file(
        &format!("{COLOR_HEADER}{RGB_PROPS}"),
        &[
            (&[1.0, 2.0, 3.0], [100, 150, 200]),
            (&[4.0, 5.0, 6.0], [10, 20, 30]),
            (&[7.0, 8.0, 9.0], [50, 60, 70]),
        ],
    );
    std::fs::File::create(&test_file)
        .unwrap()
        .write_all(&bytes)
        .unwrap();

    let mesh = read_ply_mesh_path(&test_file).unwrap().mesh;
    assert_eq!(mesh.vertices.len(), 3);
    assert_eq!(mesh.faces.len(), 1);
    assert_eq!(mesh.vertices[0].x, 1.0f32);
    assert_eq!(mesh.vertices[0].y, 2.0f32);
    assert_eq!(mesh.vertices[0].z, 3.0f32);
    assert_eq!(mesh.vertices[0].r, 100);
    assert_eq!(mesh.vertices[0].g, 150);
    assert_eq!(mesh.vertices[0].b, 200);
    assert_eq!(mesh.vertices[1].x, 4.0f32);
    assert_eq!(mesh.vertices[1].r, 10);
    assert_eq!(mesh.vertices[1].g, 20);
    assert_eq!(mesh.vertices[1].b, 30);
    assert_eq!(mesh.vertices[2].x, 7.0f32);
    assert_eq!(mesh.vertices[2].r, 50);
    assert_eq!(mesh.vertices[2].g, 60);
    assert_eq!(mesh.vertices[2].b, 70);
}

/// The colors both extra-properties tests read past the normals.
fn assert_extra_properties_mesh(mesh: &PlyMesh) {
    assert_eq!(mesh.vertices.len(), 3);
    // Positions should be read correctly.
    assert_eq!(mesh.vertices[0].x, 1.0f32);
    assert_eq!(mesh.vertices[0].y, 2.0f32);
    assert_eq!(mesh.vertices[0].z, 3.0f32);
    assert_eq!(mesh.vertices[1].x, 4.0f32);
    assert_eq!(mesh.vertices[2].x, 7.0f32);
    // Colors should be read correctly despite extra normal properties.
    assert_eq!(mesh.vertices[0].r, 255);
    assert_eq!(mesh.vertices[0].g, 128);
    assert_eq!(mesh.vertices[0].b, 64);
    assert_eq!(mesh.vertices[1].r, 32);
    assert_eq!(mesh.vertices[1].g, 64);
    assert_eq!(mesh.vertices[1].b, 96);
    assert_eq!(mesh.vertices[2].r, 10);
    assert_eq!(mesh.vertices[2].g, 20);
    assert_eq!(mesh.vertices[2].b, 30);
}

#[test]
fn ply_read_text_ply_mesh_with_extra_properties() {
    let test_file = create_test_dir().join("mesh_extra.ply");
    // Write a PLY mesh with normals and colors (normals should be skipped).
    let text = format!(
        "ply\nformat ascii 1.0\nelement vertex 3\n{COLOR_HEADER}{NORMAL_PROPS}{RGB_PROPS}\
         {FACE_PROPS}1 2 3 0.1 0.2 0.3 255 128 64\n4 5 6 0.4 0.5 0.6 32 64 96\n\
         7 8 9 0.7 0.8 0.9 10 20 30\n3 0 1 2\n"
    );
    std::fs::write(&test_file, text).unwrap();
    let mesh = read_ply_mesh_path(&test_file).unwrap().mesh;
    assert_extra_properties_mesh(&mesh);
}

#[test]
fn ply_read_binary_ply_mesh_with_extra_properties() {
    let test_file = create_test_dir().join("mesh_extra_bin.ply");
    // Write a binary PLY mesh with normals + colors.
    let bytes = binary_mesh_file(
        &format!("{COLOR_HEADER}{NORMAL_PROPS}{RGB_PROPS}"),
        &[
            (&[1.0, 2.0, 3.0, 0.1, 0.2, 0.3], [255, 128, 64]),
            (&[4.0, 5.0, 6.0, 0.4, 0.5, 0.6], [32, 64, 96]),
            (&[7.0, 8.0, 9.0, 0.7, 0.8, 0.9], [10, 20, 30]),
        ],
    );
    std::fs::write(&test_file, bytes).unwrap();
    let mesh = read_ply_mesh_path(&test_file).unwrap().mesh;
    assert_extra_properties_mesh(&mesh);
}

#[test]
fn ply_has_ply_mesh_faces_with_mesh() {
    let test_file = create_test_dir().join("mesh.ply");
    write_text_ply_mesh_path(&test_file, &PlyTexturedMesh::from(triangle_mesh())).unwrap();
    assert!(has_ply_mesh_faces_path(&test_file).unwrap());
}

#[test]
fn ply_has_ply_mesh_faces_with_point_cloud() {
    let test_file = create_test_dir().join("points.ply");
    let points = vec![PlyPoint::default()];
    write_text_ply_points_path(&test_file, &points, false, false).unwrap();
    assert!(!has_ply_mesh_faces_path(&test_file).unwrap());
}

#[test]
fn ply_has_ply_mesh_faces_with_zero_faces() {
    let test_file = create_test_dir().join("mesh_no_faces.ply");
    write_text_ply_mesh_path(&test_file, &PlyTexturedMesh::from(PlyMesh::default())).unwrap();
    assert!(!has_ply_mesh_faces_path(&test_file).unwrap());
}
