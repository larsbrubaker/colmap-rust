// Rust-only tests of colmap_rust::util::ply: the exact bytes the text writer produces (Tier A,
// the `%.6g` of a classic-locale ostream) and the truncated-texcoord check of
// docs/CPP_DIVERGENCES.md entry 280, over in-memory streams.

use std::io::Cursor;

use colmap_rust::util::ply::*;

#[test]
fn rust_only_text_points_bytes() {
    let point = PlyPoint {
        x: 1.5,
        y: -2.25,
        z: 1e-7,
        nx: 0.15,
        ny: 0.25,
        nz: 0.35,
        r: 15,
        g: 25,
        b: 255,
    };
    let mut out = Vec::new();
    write_text_ply_points(&mut out, &[point], true, true).unwrap();
    let expected = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n\
                    property float y\nproperty float z\nproperty float nx\nproperty float ny\n\
                    property float nz\nproperty uchar red\nproperty uchar green\n\
                    property uchar blue\nend_header\n1.5 -2.25 1e-07 0.15 0.25 0.35 15 25 255\n";
    assert_eq!(String::from_utf8(out).unwrap(), expected);
}

#[test]
fn rust_only_text_textured_mesh_bytes() {
    let mesh = PlyTexturedMesh {
        mesh: PlyMesh {
            vertices: vec![
                PlyMeshVertex::new(0.0, 0.0, 0.0),
                PlyMeshVertex::new(1.0, 0.0, 0.0),
                PlyMeshVertex::new(0.0, 1.0, 0.0),
            ],
            faces: vec![PlyMeshFace::new(0, 1, 2)],
        },
        face_uvs: vec![0.0, 0.0, 1.0, 0.0, 0.25, 0.75],
        texture_file: "tex.png".into(),
    };
    let mut out = Vec::new();
    write_text_ply_mesh(&mut out, &mesh).unwrap();
    let expected = "ply\nformat ascii 1.0\ncomment TextureFile tex.png\nelement vertex 3\n\
                    property float x\nproperty float y\nproperty float z\nelement face 1\n\
                    property list uchar int vertex_indices\nproperty list uchar float texcoord\n\
                    end_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2 6 0 0 1 0 0.25 0.75\n";
    assert_eq!(String::from_utf8(out).unwrap(), expected);
}

#[test]
fn rust_only_mismatched_uv_count_fails() {
    let mut mesh = PlyTexturedMesh::from(PlyMesh {
        vertices: vec![PlyMeshVertex::default(); 3],
        faces: vec![PlyMeshFace::new(0, 1, 2)],
    });
    mesh.face_uvs = vec![0.0; 5];
    let error = write_text_ply_mesh(&mut Vec::new(), &mesh).unwrap_err();
    assert!(error
        .message()
        .contains("Expected 6 UV coordinates per face"));
}

#[test]
fn rust_only_truncated_binary_texcoords_fail() {
    let mesh = PlyTexturedMesh {
        mesh: PlyMesh {
            vertices: vec![PlyMeshVertex::default(); 3],
            faces: vec![PlyMeshFace::new(0, 1, 2)],
        },
        face_uvs: vec![0.5; 6],
        texture_file: String::new(),
    };
    let mut bytes = Vec::new();
    write_binary_ply_mesh(&mut bytes, &mesh).unwrap();
    assert_eq!(
        read_ply_mesh(&mut Cursor::new(&bytes))
            .unwrap()
            .face_uvs
            .len(),
        6
    );
    // Cut before the texcoord count, and after two of the six UVs (the last face).
    for cut in [bytes.len() - 25, bytes.len() - 16] {
        let error = read_ply_mesh(&mut Cursor::new(&bytes[..cut])).unwrap_err();
        assert!(
            error
                .message()
                .contains("Unexpected end of PLY file at face 0"),
            "{}",
            error.message()
        );
    }
}

#[test]
fn rust_only_missing_xyz_fails() {
    let text = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n1\n";
    let error = read_ply(&mut Cursor::new(text)).unwrap_err();
    assert!(error
        .message()
        .contains("Invalid PLY file format: x, y, z properties missing"));
}

#[test]
fn rust_only_has_faces_skips_malformed_count() {
    let text = "ply\nformat ascii 1.0\nelement face abc\nelement face 2\nend_header\n";
    assert!(has_ply_mesh_faces(&mut Cursor::new(text)).unwrap());
}
