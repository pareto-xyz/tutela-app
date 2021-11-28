import React from 'react';
import AgnosticTable from '../components/AgnosticTable';

export default function TornadoInfo({ data }) {
    return (
        <div className="tornado-info ">
            <div className="panel-sub">tornado cash statistics</div>
            <div className="panel-title">
                TORNADO CASH INFO
            </div>
            <div className="panel-sub">
                Compromised addresses are deposit addresses that are deemed revealing of identity through Tutela heuristics.
            </div>
            {data.address && <AgnosticTable keyValues={Object.entries(data.address)} />}
            {data.cluster && <AgnosticTable keyValues={Object.entries(data.cluster)} />}
        </div>

    );
}