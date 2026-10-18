//! `hamiltonian`: Jesse v1.3 Hamiltonian-Fourier Cognitive Prosody, $D=64$ Phase Binding & 8-Axis Emotions.
//!
//! - **Cloud WebSocket Execution (`wss://api.solidsf.com/v1/ssftalk/ws`)**: Use [`dispatch_over_ws`]
//!   with [`crate::cloud_ws::SsfTalkWsClient`] (`WsEngine::HamiltonianProsodyDirector`).
//! - **Sealed Binary Vault (`assets/ssftalk_moat_vault.bin`)**: Local benchmark & offline execution
//!   is unpacked at compile time from the Zobrist-Feistel encrypted vault (`build.rs`).

include!(concat!(env!("OUT_DIR"), "/vault_hamiltonian.rs"));

/// Dispatch a `HamiltonianProsodyDirector` cognitive trajectory & 8-axis prosody coupling request
/// over the active binary WebSocket connection (`wss://api.solidsf.com/v1/ssftalk/ws`).
pub fn dispatch_over_ws(
    client: &mut crate::cloud_ws::SsfTalkWsClient,
    text: &str,
    total_frames: usize,
) -> crate::Result<crate::cloud_ws::WsRpcResponse> {
    let req = crate::cloud_ws::WsRpcRequest {
        engine: crate::cloud_ws::WsEngine::HamiltonianProsodyDirector,
        operation: "hamiltonian_cognitive_prosody_trajectory".to_string(),
        payload_json: format!(
            "{{\"text\":{:?},\"total_frames\":{total_frames}}}",
            text
        ),
    };
    client.call(&req)
}
