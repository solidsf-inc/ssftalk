//! `duplex`: Full-Duplex Conversational AI Voice Rooms (`JAUD` wire protocol) & 4-Gate Von Neumann Minimax Speech Jury.
//!
//! - **Cloud WebSocket Execution (`wss://api.solidsf.com/v1/ssftalk/ws`)**: Use [`dispatch_over_ws`]
//!   with [`crate::cloud_ws::SsfTalkWsClient`] (`WsEngine::DuplexDiarizer4D`).
//! - **Sealed Binary Vault (`assets/ssftalk_moat_vault.bin`)**: Local benchmark & offline execution
//!   is unpacked at compile time from the Zobrist-Feistel encrypted vault (`build.rs`).

include!(concat!(env!("OUT_DIR"), "/vault_duplex.rs"));

/// Dispatch a `DuplexDiarizer4D` full-duplex room diarization & floor-control request over the
/// active binary WebSocket connection (`wss://api.solidsf.com/v1/ssftalk/ws`).
pub fn dispatch_over_ws(
    client: &mut crate::cloud_ws::SsfTalkWsClient,
    room_id: &str,
    participants: usize,
) -> crate::Result<crate::cloud_ws::WsRpcResponse> {
    let req = crate::cloud_ws::WsRpcRequest {
        engine: crate::cloud_ws::WsEngine::DuplexDiarizer4D,
        operation: "duplex_room_diarize_turn".to_string(),
        payload_json: format!(
            "{{\"room_id\":{:?},\"participants\":{participants}}}",
            room_id
        ),
    };
    client.call(&req)
}
