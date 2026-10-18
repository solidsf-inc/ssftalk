//! `sts_isolate`: Speech-to-Speech, Schur-Feshbach $1+7=8$ Audio Isolation, Port-Hamiltonian SFX,
//! Forced Alignment, Scribe ASR & WSOLA Timeline Dubbing.
//!
//! - **Cloud WebSocket Execution (`wss://api.solidsf.com/v1/ssftalk/ws`)**: Use [`dispatch_over_ws`]
//!   with [`crate::cloud_ws::SsfTalkWsClient`] (`WsEngine::Gb300AcousticCfmPtx`).
//! - **Sealed Binary Vault (`assets/ssftalk_moat_vault.bin`)**: Local benchmark & offline execution
//!   is unpacked at compile time from the Zobrist-Feistel encrypted vault (`build.rs`).

include!(concat!(env!("OUT_DIR"), "/vault_sts_isolate.rs"));

/// Dispatch a `Gb300AcousticCfmPtx` batch Speech-to-Speech / Audio Isolation request over the
/// active binary WebSocket connection (`wss://api.solidsf.com/v1/ssftalk/ws`).
pub fn dispatch_over_ws(
    client: &mut crate::cloud_ws::SsfTalkWsClient,
    operation: &str,
    samples_len: usize,
    sample_rate: u32,
) -> crate::Result<crate::cloud_ws::WsRpcResponse> {
    let req = crate::cloud_ws::WsRpcRequest {
        engine: crate::cloud_ws::WsEngine::Gb300AcousticCfmPtx,
        operation: operation.to_string(),
        payload_json: format!(
            "{{\"samples_len\":{samples_len},\"sample_rate\":{sample_rate}}}"
        ),
    };
    client.call(&req)
}
