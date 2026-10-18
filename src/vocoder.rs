//! `vocoder`: Phase-Continuous Rosenberg Glottal + 8-Subband iSTFT Vocoder, Partition-of-Unity Streaming & WSOLA.
//!
//! - **Cloud WebSocket Execution (`wss://api.solidsf.com/v1/ssftalk/ws`)**: Use [`dispatch_over_ws`]
//!   with [`crate::cloud_ws::SsfTalkWsClient`] (`WsEngine::Articulate150mNeural`).
//! - **Sealed Binary Vault (`assets/ssftalk_moat_vault.bin`)**: Local benchmark & offline execution
//!   is unpacked at compile time from the Zobrist-Feistel encrypted vault (`build.rs`).

include!(concat!(env!("OUT_DIR"), "/vault_vocoder.rs"));

/// Dispatch an `Articulate150mNeural` synthesis request over the active binary
/// WebSocket connection (`wss://api.solidsf.com/v1/ssftalk/ws`).
pub fn dispatch_over_ws(
    client: &mut crate::cloud_ws::SsfTalkWsClient,
    voice_id: &str,
    text: &str,
    sample_rate: u32,
) -> crate::Result<crate::cloud_ws::WsRpcResponse> {
    let req = crate::cloud_ws::WsRpcRequest {
        engine: crate::cloud_ws::WsEngine::Articulate150mNeural,
        operation: "articulate_150m_neural_tts".to_string(),
        payload_json: format!(
            "{{\"voice_id\":{:?},\"text\":{:?},\"sample_rate\":{sample_rate}}}",
            voice_id, text
        ),
    };
    client.call(&req)
}
