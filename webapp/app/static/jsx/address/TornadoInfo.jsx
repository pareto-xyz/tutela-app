import React from 'react';
import AgnosticTable from '../components/AgnosticTable';

export default function TornadoInfo({ data, aliases }) {
    return (
        <div className="tornado-info ">
            <div className="panel-sub">about your input</div>
            <div className="panel-title">
                TORNADO CASH STATISTICS
            </div>
            <div className="panel-sub">
            This shows Tornado Cash transactions by your input address and other clustered addresses.
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

                        <AgnosticTable aliases={aliases} keyValues={Object.entries(data.summary.cluster)} />
                    </div>
                }

            </div>

        </div>

    );
}