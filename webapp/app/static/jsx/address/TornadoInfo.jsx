import React from 'react';
import AgnosticTable from '../components/AgnosticTable';

export default function TornadoInfo({ data, aliases }) {
    return (
        <div className="col-md-12 col-lg-6">
            <div className="row tornado-row">
                <div className="col-12">
                    <div className="tornado-info">
                        <div className="panel-sub col-12">about your input</div>
                        <div className="panel-title col-12">
                            TORNADO CASH STATISTICS
                        </div>
                        <div className="panel-sub col-12">
                            This shows Tornado Cash transactions by your input address.
                        </div>
                        <div className="two-tables col-12">
                            {data && data.summary && data.summary.address && <div className="row">
                                <div className="table-title col-12">by address </div>
                                <AgnosticTable aliases={aliases} keyValues={data.summary.address} />
                            </div>
                            }
                            {data && data.summary && data.summary.cluster &&
                                <div>
                                    <div className="table-title">by cluster </div>

                                    <AgnosticTable aliases={aliases} keyValues={data.summary.cluster} />
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
                </div>
            </div>
        </div>
    );
}