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
import exampleResponse from '../../data/plot';

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
        console.log(e)
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
                            <option value='1yr'>
                                past year
                            </option>
                            <option value='3mth'>
                                past 3 months
                            </option>
                            <option value='6mth'>
                                past 6 months
                            </option>
                            <option value='3yr'>
                                past 3 years
                            </option>
                            <option value='5yr'>
                                past 5 years
                            </option>
                        </Form.Select>
                    </div>
                </div>
                <div className="panel-sub col-12">
                    This shows when you committed Ethereum or Tornado Cash reveals over the selected time period.
                    <ResponsiveContainer width="100%" height={600} >
                        <BarChart onClick={onClickWeek}
                            data={plotData}
                            margin={{ top: 5, right: 0, left: -30, bottom: 30 }}>
                            <CartesianGrid />
                            <XAxis dataKey="name" />
                            <YAxis />
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