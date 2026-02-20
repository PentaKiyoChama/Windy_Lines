;// __TPL_MATCH_NAME__.i
;// Windows Linker Module Definition
;// GPU エントリーポイントのエクスポート

EXPORTS
    ; CPU エントリーポイント
    PluginMain

    ; GPU エントリーポイント（PrGPUFilterModule 経由）
    xGPUFilterEntry
