// Reconstruction pipeline stages and their status, as shown in the app's "Pipeline" panel.
// The panel (`pipeline_panel.rs`) reads this every layout, so a phase that ports a stage just
// changes the entry here (or the runner updates it at run time) and the UI follows.
//
// Stage list mirrors COLMAP's automatic reconstruction flow (`controllers/
// automatic_reconstruction.cc`): images → feature extraction → matching → incremental mapping
// → bundle adjustment → PatchMatch stereo → fusion → meshing.

/// One stage of the photos → mesh pipeline, in execution order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    Images,
    Features,
    Matching,
    SparseReconstruction,
    BundleAdjustment,
    DensePatchMatch,
    Fusion,
    Meshing,
}

impl Stage {
    /// Every stage, in pipeline order.
    pub const ALL: [Stage; 8] = [
        Stage::Images,
        Stage::Features,
        Stage::Matching,
        Stage::SparseReconstruction,
        Stage::BundleAdjustment,
        Stage::DensePatchMatch,
        Stage::Fusion,
        Stage::Meshing,
    ];

    /// Display name in the Pipeline panel.
    pub fn title(self) -> &'static str {
        match self {
            Stage::Images => "Images",
            Stage::Features => "Features",
            Stage::Matching => "Matching",
            Stage::SparseReconstruction => "Sparse reconstruction",
            Stage::BundleAdjustment => "Bundle adjustment",
            Stage::DensePatchMatch => "Dense (PatchMatch)",
            Stage::Fusion => "Fusion",
            Stage::Meshing => "Meshing",
        }
    }

    /// Stable identifier used for widget ids (`stage-<key>`) and tests.
    pub fn key(self) -> &'static str {
        match self {
            Stage::Images => "images",
            Stage::Features => "features",
            Stage::Matching => "matching",
            Stage::SparseReconstruction => "sparse",
            Stage::BundleAdjustment => "bundle-adjustment",
            Stage::DensePatchMatch => "patch-match",
            Stage::Fusion => "fusion",
            Stage::Meshing => "meshing",
        }
    }

    /// Widget id of this stage's row in the Pipeline panel.
    pub fn row_id(self) -> String {
        format!("stage-{}", self.key())
    }
}

/// Where a stage stands.
#[derive(Clone, Debug, PartialEq)]
pub enum StageStatus {
    /// The library code for this stage has not been ported to Rust yet.
    NotYetPorted,
    /// Ported and available, not run on the current project.
    Ready,
    /// Running; progress in [0, 1].
    Running { progress: f32 },
    /// Finished successfully on the current project.
    Done,
    /// Finished with an error.
    Failed(String),
}

impl StageStatus {
    /// Text shown next to the stage name.
    pub fn label(&self) -> String {
        match self {
            StageStatus::NotYetPorted => "Not yet ported".to_string(),
            StageStatus::Ready => "Ready".to_string(),
            StageStatus::Running { progress } => {
                format!("Running {:.0}%", (progress.clamp(0.0, 1.0) * 100.0).round())
            }
            StageStatus::Done => "Done".to_string(),
            StageStatus::Failed(reason) => format!("Failed: {reason}"),
        }
    }
}

/// Status of every stage.
#[derive(Clone, Debug, PartialEq)]
pub struct PipelineStatus {
    statuses: [StageStatus; 8],
}

impl Default for PipelineStatus {
    /// The truthful state of the port today: no stage has library code yet.
    fn default() -> Self {
        Self {
            statuses: std::array::from_fn(|_| StageStatus::NotYetPorted),
        }
    }
}

impl PipelineStatus {
    /// Discriminants follow declaration order, which is `Stage::ALL` order.
    fn index(stage: Stage) -> usize {
        stage as usize
    }

    pub fn status(&self, stage: Stage) -> &StageStatus {
        &self.statuses[Self::index(stage)]
    }

    pub fn set_status(&mut self, stage: Stage, status: StageStatus) {
        self.statuses[Self::index(stage)] = status;
    }

    /// `(stage, status)` in pipeline order.
    pub fn iter(&self) -> impl Iterator<Item = (Stage, &StageStatus)> {
        Stage::ALL.iter().copied().zip(self.statuses.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_marks_every_stage_not_yet_ported() {
        let p = PipelineStatus::default();
        assert_eq!(p.iter().count(), 8);
        assert!(p.iter().all(|(_, s)| *s == StageStatus::NotYetPorted));
    }

    #[test]
    fn set_status_changes_only_that_stage() {
        let mut p = PipelineStatus::default();
        p.set_status(Stage::Matching, StageStatus::Done);
        assert_eq!(*p.status(Stage::Matching), StageStatus::Done);
        assert_eq!(*p.status(Stage::Features), StageStatus::NotYetPorted);
    }

    #[test]
    fn labels_and_keys_are_distinct() {
        let mut keys: Vec<_> = Stage::ALL.iter().map(|s| s.key()).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), Stage::ALL.len());
        assert_eq!(
            StageStatus::Running { progress: 0.456 }.label(),
            "Running 46%"
        );
    }
}
