import React from 'react';
import AgnosticTable from '../components/AgnosticTable';

export default function TornadoInfo({ data, aliases }) {
    return (
        <div className="tornado-info ">
            <div className="panel-sub">tornado cash statistics</div>
            <div className="panel-title">
                ABOUT YOUR INPUT
            </div>
            <div className="panel-sub">
            This shows Tornado Cash transactions by your input address and addresses it is clustered with.
            </div>
            <div className="two-tables">
                {data && data.summary && data.summary.address && <div>
                    <div className="table-title">by address </div>
                    <AgnosticTable aliases={aliases} keyValues={Object.entries(data.summary.address)} />
                </div>
                }
                {data && data.summary && data.summary.cluster &&
                    <div>
                        <div className="table-title">by cluster </div>

                        <AgnosticTable keyValues={Object.entries(data.summary.cluster)} />
                    </div>
                }

            </div>

        </div>

    );
}