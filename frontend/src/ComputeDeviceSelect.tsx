import { useEffect, useState } from 'react';
import { api } from './api';

export function ComputeDeviceSelect({
  value,
  onChange,
  onError,
  label = 'Compute device',
}: {
  value: string;
  onChange: (device: string) => void;
  onError: (message: string) => void;
  label?: string;
}) {
  const [devices, setDevices] = useState<{ id: string; label: string }[]>([]);
  useEffect(() => {
    let active = true;
    void api<{ devices: { id: string; label: string }[] }>('/api/catalog')
      .then((catalog) => {
        if (active) setDevices(catalog.devices.filter((device) => !device.id.startsWith('cpu')));
      })
      .catch((error) => {
        if (active) onError(String(error));
      });
    return () => {
      active = false;
    };
  }, [onError]);

  return (
    <label>
      Compute device
      <select aria-label={label} value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="cpu">CPU</option>
        {value !== 'cpu' && !devices.some((device) => device.id === value) && (
          <option value={value} disabled>
            {value}
          </option>
        )}
        {devices.map((device) => (
          <option key={device.id} value={device.id}>
            {device.label}
          </option>
        ))}
      </select>
    </label>
  );
}
