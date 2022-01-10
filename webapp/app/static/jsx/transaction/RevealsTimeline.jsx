import React, { useEffect, useState } from 'react';
import {
    BarChart, Bar, XAxis, YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer
} from 'recharts';
import { Form } from 'react-bootstrap';
import { getApi } from '../../js/utils';
//import exampleResponse from '../../data/plot';

const TIME_PERIODS = {
    '3mth': 'past 3 months',
    '6mth': 'past 6 months',
    '1yr': 'past year',
    '3yr': 'past 3 years',
    '5yr': 'past 5 years'
}

export default function RevealTimeline({ addr, loadNewData, aliases }) {
    // const {start_date, end_date, counts } = plotData;
    const [plotData, setPlotData] = useState([]);
    const [plotWindow, setPlotWindow] = useState('1yr');
    // const [selectedWeek, setSelectedWeek] = useState(''); //the name

    useEffect(() => {
        onSelectWindow('1yr');
    }, [])

    function onSelectWindow(windowOption) {
        setPlotWindow(windowOption);
        getApi(`/plot/transaction?address=${addr}&window=${windowOption}`, response => {
            // response = exampleResponse;
            const { query, data, success } = response.data;
            if (success === 1) {
                setPlotData(data);
            }
        })
    }

    function onClickWeek(e) {
        const { activeLabel, activeTooltipIndex } = e;
        const [startDate, endDate] = activeLabel.split('-');
        loadNewData({
            address: addr,
            start_date: startDate,
            end_date: endDate
        });
        // setSelectedWeek(activeLabel);
        // console.log($('path.recharts-tooltip-cursor').addClass())
    }

    return (
        <div className="col-md-12 col-lg-6">
            <div className="tornado-info col-12">
                <div className="panel-sub col-12">about your input</div>
                <div className="panel-title   row ">
                    <div className="col-6 title-text">
                        TIMELINE OF REVEALS

                    </div>
                    <div className="col-6 select-window">
                        <Form.Select value={plotWindow} id="window" className="select-window-button" onChange={e => onSelectWindow(e.target.value)}>
                            {Object.entries(TIME_PERIODS).map(([code, english]) => <option value={code}> {english} </option>)}
                        </Form.Select>
                    </div>
                </div>
                <div className="panel-sub col-12">
                    This shows reveals in the {TIME_PERIODS[plotWindow]}. Click on a bar in the chart to show the reveal transactions below.
                    <ResponsiveContainer width="100%" height={600} >
                        <BarChart onClick={onClickWeek}
                            data={plotData}
                            margin={{ top: 5, right: 0, left: -20, bottom: 30 }}>
                            <CartesianGrid />
                            <XAxis dataKey="name" interval='preserveStartEnd'/>
                            <YAxis label={{ value: '# of reveals', fill: 'white', angle: -90, offset: 30, position: 'insideLeft' }} className="yaxis" />
                            <Tooltip
                                contentStyle={{ 'backgroundColor': '#404040', 'border': 'transparent', 'borderRadius': '5px' }} />
                            <Bar name={aliases.deposit_address_reuse || 'deposit_address_reuse'} dataKey="deposit_address_reuse" stackId="a" fill="#8884d8" />
                            <Bar name={aliases.unique_gas_price || 'unique_gas_price'} dataKey="unique_gas_price" stackId="a" fill="pink" />
                            <Bar name={aliases.multi_denomination || 'multi_denomination'} dataKey="multi_denomination" stackId="a" fill="violet" />
                            <Bar name={aliases.linked_transaction || 'linked_transaction'} dataKey="linked_transaction" stackId="a" fill="#82ca9d" />
                            <Bar name={aliases.address_match || 'address_match'} dataKey="address_match" stackId="a" fill="lightblue" />
                            <Bar name={aliases.torn_mine || 'torn_mine'} dataKey="torn_mine"
                                // background={({ name }) => (
                                //     <div style={{ height: '100%', width: '50px', fill: name === selectedWeek ? '#ee11bb' : '#ff0033' }}></div>
                                // )}
                                stackId="a" fill="white" />

                            <Legend verticalAlign="bottom" height={5} />

                        </BarChart>

                    </ResponsiveContainer>

                </div>
            </div>
        </div>
    )
};