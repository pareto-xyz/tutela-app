import React from 'react';
import AgnosticTable from '../components/AgnosticTable';

export default function TpoolStats({ data, aliases }) {
    return (
        <div className="tornado-info ">
            <div className="panel-title">
                TORNADO CASH ANONYMITY SET
            </div>
            {data &&
                <div>
                    <AgnosticTable aliases={aliases} keyValues={data} />
                </div>
            }
        </div>
    )
}