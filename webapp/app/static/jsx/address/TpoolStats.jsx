import React from 'react';
import AgnosticTable from '../components/AgnosticTable';

export default function TpoolStats({ data, aliases }) {
    return (
        <div className="col-md-12 col-lg-6">
            <div className="tornado-info row">
                <div className="panel-title col-12">
                    TORNADO CASH ANONYMITY SET
                </div>
                {data &&
                    <div>
                        <AgnosticTable aliases={aliases} keyValues={data} />
                    </div>
                }
            </div>
        </div>
    )
}